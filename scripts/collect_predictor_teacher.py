#!/usr/bin/env python3
"""Collect current wall-x VisPruner teacher scores and layer-8 features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image


def sample_image(sample: dict, key: str) -> Image.Image:
    image = sample[key]
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = (image.clamp(0, 1) * 255).to(torch.uint8).numpy()
    return Image.fromarray(image).convert("RGB")


class LocalVideoSamples:
    """Read the local LeRobot 3.0 parquet/video layout without Hub metadata."""

    def __init__(self, root: str, image_key: str):
        import pandas as pd

        self.root = Path(root)
        self.image_key = image_key
        self.files = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        self.rows = []
        self.base = {}
        for path in self.files:
            video = self.root / "videos" / self.image_key / path.parent.name / path.name.replace(".parquet", ".mp4")
            if not video.exists():
                # Some local exports contain state-only parquet shards.  They
                # cannot provide visual teacher examples, so skip them.
                continue
            frame = pd.read_parquet(path, columns=["index", "episode_index"])
            values = [
                (int(row.index), int(row.episode_index))
                for row in frame.itertuples(index=False)
            ]
            self.base[path] = min(i for i, _episode in values)
            self.rows.extend((path, i, episode) for i, episode in values)
        self.rows.sort(key=lambda item: item[1])
        self._active_path = None
        self._active_container = None
        self._active_frames = None
        self._active_frame_index = -1

    def __len__(self):
        return len(self.rows)

    def image(self, index: int) -> Image.Image:
        path, global_index, _episode = self.rows[index]
        import av

        video = self.root / "videos" / self.image_key / path.parent.name / path.name.replace(".parquet", ".mp4")
        target = global_index - self.base[path]
        if path != self._active_path or target < self._active_frame_index:
            if self._active_container is not None:
                self._active_container.close()
            self._active_container = av.open(str(video))
            self._active_frames = self._active_container.decode(self._active_container.streams.video[0])
            self._active_path = path
            self._active_frame_index = -1
        while self._active_frame_index < target:
            try:
                frame = next(self._active_frames)
            except StopIteration as exc:
                raise IndexError(f"video frame {global_index} not found in {video}") from exc
            self._active_frame_index += 1
        return frame.to_image().convert("RGB")

    def episode(self, index: int) -> int:
        return int(self.rows[index][2])


def split_sample_indices(
    dataset: LocalVideoSamples,
    *,
    num_samples: int,
    split: str,
    seed: int,
) -> list[int]:
    """Select a global sample budget, then assign samples by episode split.

    The same call with ``split=train``, ``val`` and ``test`` produces disjoint
    subsets whose combined size is approximately ``num_samples``. This keeps
    the requested 5000 moments as a total budget rather than 5000 per split.
    """

    all_indices = np.arange(len(dataset), dtype=np.int64)
    episodes = np.array(sorted({dataset.episode(int(i)) for i in all_indices}))
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_test = max(1, int(round(len(episodes) * 0.1)))
    n_val = max(1, int(round(len(episodes) * 0.1)))
    test_eps = set(int(x) for x in episodes[:n_test])
    val_eps = set(int(x) for x in episodes[n_test : n_test + n_val])
    if split not in {"train", "val", "test", "all"}:
        raise ValueError(f"unknown split {split!r}")
    if num_samples <= 0 or num_samples >= len(all_indices):
        sampled = [int(i) for i in all_indices]
    else:
        positions = np.linspace(0, len(all_indices) - 1, num_samples, dtype=int)
        sampled = [int(all_indices[int(pos)]) for pos in positions]
    if split == "all":
        selected = sampled
    elif split == "train":
        selected = [i for i in sampled if dataset.episode(i) not in test_eps | val_eps]
    elif split == "val":
        selected = [i for i in sampled if dataset.episode(i) in val_eps]
    else:
        selected = [i for i in sampled if dataset.episode(i) in test_eps]
    if not selected:
        raise ValueError(f"episode split {split!r} has no sampled video frames")
    return selected


def save_shard(records: list[dict], output_dir: Path, split: str, shard_idx: int, meta: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"teacher_{split}_{shard_idx:05d}.pt"
    torch.save({"meta": meta, "records": records}, path)
    print(f"[PREDICTOR TEACHER] saved {len(records)} records to {path}", flush=True)
    return path


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    from wall_x._vendor.harrix.adapters import qwen_vlact  # noqa: F401
    from wall_x._vendor.harrix.adapters.registry import build_adapter
    from wall_x._vendor.harrix.eval_config import EvalConfig, LiberoEnvParams, autofill_from_checkpoint
    from wall_x.model.qact.qwen2_5.vispruner.hook import pop_scores
    from wall_x.model.qact.qwen2_5.vispruner.planner import _align_patch_scores
    from wall_x.model.qact.qwen2_5.vispruner.scoring import reduce_scores
    from wall_x.model.qact.qwen2_5.vispruner.algorithms import select_indices

    cfg = EvalConfig()
    cfg.model.checkpoint_path = args.checkpoint_path
    cfg.model.train_config_path = args.train_config_path
    cfg.model.cam_names = list(args.image_key)
    cfg.model.action_mode = "flow"
    cfg.env.libero = LiberoEnvParams(num_trials_per_task=1, task_indices=[0])
    cfg = autofill_from_checkpoint(cfg)
    adapter = build_adapter(cfg)
    device = torch.device(adapter._device)
    image_keys = list(args.image_key)
    datasets = {key: LocalVideoSamples(args.dataset_root, key) for key in image_keys}
    dataset_size = min(len(dataset) for dataset in datasets.values())
    if any(len(dataset) != dataset_size for dataset in datasets.values()):
        raise ValueError("camera video row counts do not match")
    indices = split_sample_indices(
        datasets[image_keys[0]],
        num_samples=args.num_samples,
        split=args.split,
        seed=args.split_seed,
    )
    indices = [
        i for i in indices if i >= args.start_index and (args.end_index is None or i < args.end_index)
    ]
    output = []
    shard_paths = []
    meta = {
        "feature_source": "early_hidden", "early_layer": args.early_layer,
        "score_layer": args.score_layer, "score_reduce": "attn_max",
        "repo_id": args.repo_id, "split": args.split,
        "num_samples_requested": args.num_samples,
    }
    total = len(indices) * len(image_keys)
    count = 0
    for image_key in image_keys:
      dataset = datasets[image_key]
      for dataset_index in indices:
        count += 1
        image = dataset.image(dataset_index)
        inputs = adapter._processor.image_processor(images=[[image]], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=adapter._model.visual.dtype)
        grid = inputs["image_grid_thw"].to(device)
        embeds, early = adapter._model.visual(
            pixel_values,
            grid_thw=grid,
            return_early_features=True,
            predictor_early_layer=args.early_layer,
        )
        scores = pop_scores(adapter._model.visual)
        layer = scores.get(args.score_layer)
        if not layer or layer.get("attn") is None:
            raise RuntimeError(f"teacher attention scores missing at layer {args.score_layer}")
        raw = [reduce_scores("attn_max", attn=part) for part in layer["attn"]]
        aligned = _align_patch_scores(
            adapter._model.visual,
            grid,
            raw,
            adapter._model.config.vision_config.spatial_merge_size,
        )
        teacher = torch.cat(aligned).float().cpu()
        features = early.float().cpu()
        if features.shape[0] != teacher.shape[0]:
            raise RuntimeError(f"feature/teacher length mismatch: {features.shape} vs {teacher.shape}")
        teacher_keep = select_indices(
            teacher,
            embeds.float().cpu(),
            args.keep_num,
            args.important_ratio,
        )
        output.append({
            "dataset_index": dataset_index,
            "episode_index": dataset.episode(dataset_index),
            "image_key": image_key,
            "predictor_features": features,
            "image_scores": teacher,
            "teacher_keep_indices": teacher_keep,
        })
        print(f"[PREDICTOR TEACHER] {count}/{total} key={image_key} index={dataset_index} tokens={features.shape[0]}", flush=True)
        if args.shard_size > 0 and len(output) >= args.shard_size:
            shard_paths.append(save_shard(output, Path(args.output_path), args.split, len(shard_paths), meta))
            output = []
    if output:
        shard_paths.append(save_shard(output, Path(args.output_path), args.split, len(shard_paths), meta))
    meta["num_records"] = total
    manifest = Path(args.output_path) / f"manifest_{args.split}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"meta": meta, "shards": [str(p) for p in shard_paths]}, indent=2), encoding="utf-8")
    print(f"[PREDICTOR TEACHER] saved manifest {manifest}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--train-config-path", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--repo-id", default="libero")
    p.add_argument("--image-key", nargs="+", default=["observation.images.image"])
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=None)
    p.add_argument("--early-layer", type=int, default=8)
    p.add_argument("--score-layer", type=int, default=31)
    p.add_argument("--keep-num", type=int, default=20)
    p.add_argument("--important-ratio", type=float, default=0.5)
    p.add_argument("--output-path", required=True, help="Directory for sharded teacher files")
    p.add_argument("--shard-size", type=int, default=500)
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    p.add_argument("--split-seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
