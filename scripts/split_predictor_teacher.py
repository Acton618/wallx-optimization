#!/usr/bin/env python3
"""Split a collected all-record teacher set by episode into train/val/test shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_paths(root: Path) -> list[Path]:
    manifest = root / "manifest_all.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        paths = [Path(p) for p in data["shards"]]
        return [p if p.is_absolute() else root / p for p in paths]
    return sorted(root.glob("teacher_all_*.pt"))


def main(args: argparse.Namespace) -> None:
    root = Path(args.input_path)
    paths = load_paths(root)
    if not paths:
        raise FileNotFoundError(f"no teacher_all shards under {root}")
    records, meta = [], {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta.update(payload.get("meta", {}))
        records.extend(payload["records"])
    episodes = np.array(sorted({int(r["episode_index"]) for r in records}), dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(episodes)
    n_test = max(1, int(round(len(episodes) * 0.1)))
    n_val = max(1, int(round(len(episodes) * 0.1)))
    test_eps = {int(x) for x in episodes[:n_test]}
    val_eps = {int(x) for x in episodes[n_test : n_test + n_val]}
    buckets = {
        "train": [r for r in records if int(r["episode_index"]) not in test_eps | val_eps],
        "val": [r for r in records if int(r["episode_index"]) in val_eps],
        "test": [r for r in records if int(r["episode_index"]) in test_eps],
    }
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    for split, split_records in buckets.items():
        split_meta = dict(meta)
        split_meta.update({
            "split": split,
            "num_records": len(split_records),
            "episode_count": len({int(r["episode_index"]) for r in split_records}),
            "split_seed": args.seed,
        })
        shard_paths = []
        for start in range(0, len(split_records), args.shard_size):
            shard = split_records[start : start + args.shard_size]
            path = out / f"teacher_{split}_{len(shard_paths):05d}.pt"
            torch.save({"meta": split_meta, "records": shard}, path)
            shard_paths.append(str(path))
            print(f"[PREDICTOR SPLIT] saved {len(shard)} records to {path}", flush=True)
        manifest = out / f"manifest_{split}.json"
        manifest.write_text(json.dumps({"meta": split_meta, "shards": shard_paths}, indent=2), encoding="utf-8")
        print(f"[PREDICTOR SPLIT] {split}: {len(split_records)} records, {split_meta['episode_count']} episodes", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--shard-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

