#!/usr/bin/env python3
"""Evaluate predictor ranking quality on a sharded held-out teacher set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wall_x.model.qact.qwen2_5.vispruner.predictor import load_predictor


def shard_paths(path: str | Path, split: str) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    manifest = path / f"manifest_{split}.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        result = []
        for raw in data.get("shards", []):
            candidate = Path(raw)
            result.append(candidate if candidate.is_absolute() or candidate.exists() else path / candidate)
        return result
    paths = sorted(path.glob(f"teacher_{split}_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no teacher shards found in {path} for split={split}")
    return paths


def rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32)
    return ranks


def main(args: argparse.Namespace) -> None:
    paths = shard_paths(args.teacher_path, args.split)
    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    first_record = first["records"][0]
    early_layer = int(first.get("meta", {}).get("early_layer", args.early_layer))
    predictor = load_predictor(
        args.predictor_path,
        input_dim=int(first_record["predictor_features"].shape[-1]),
        early_layer=early_layer,
    ).eval()
    mse, spearman, overlap = [], [], []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for record in payload["records"]:
            features = record["predictor_features"].float()
            teacher = record["image_scores"].float().reshape(-1)
            teacher = (teacher - teacher.mean()) / (teacher.std(unbiased=False) + 1e-6)
            with torch.no_grad():
                predicted = predictor(features).float()
            mse.append(torch.mean((predicted - teacher) ** 2).item())
            x_rank, y_rank = rankdata(predicted), rankdata(teacher)
            x_rank -= x_rank.mean()
            y_rank -= y_rank.mean()
            spearman.append(float((x_rank @ y_rank) / (x_rank.norm() * y_rank.norm()).clamp_min(1e-8)))
            k = min(args.keep_num, teacher.numel())
            teacher_top = torch.topk(teacher, k).indices
            predicted_top = torch.topk(predicted, k).indices
            overlap.append(float(torch.isin(predicted_top, teacher_top).float().mean()))
    print(f"split={args.split} records={len(mse)}")
    print(f"mse={sum(mse)/len(mse):.6f}")
    print(f"spearman={sum(spearman)/len(spearman):.6f}")
    print(f"top{args.keep_num}_overlap={sum(overlap)/len(overlap):.6f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-path", required=True)
    p.add_argument("--predictor-path", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--keep-num", type=int, default=40)
    p.add_argument("--early-layer", type=int, default=16)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

