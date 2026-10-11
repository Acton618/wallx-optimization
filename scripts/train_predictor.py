#!/usr/bin/env python3
"""Train a token score predictor from sharded teacher records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wall_x.model.qact.qwen2_5.vispruner.predictor import TokenScorePredictor


def shard_paths(path: str | Path, split: str) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    manifest = path / f"manifest_{split}.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        paths = [Path(p) for p in data.get("shards", [])]
        resolved = []
        for p in paths:
            if p.is_absolute() or p.exists():
                resolved.append(p)
            else:
                resolved.append(path / p)
        return resolved
    paths = sorted(path.glob(f"teacher_{split}_*.pt"))
    if not paths and split == "all":
        paths = sorted(path.glob("teacher_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no teacher shards found in {path} for split={split}")
    return paths


def load_shard(path: Path) -> tuple[dict, list[dict]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("meta", {}), payload["records"]


def target_for(record: dict, target_kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = record["predictor_features"].float()
    y = record["image_scores"].float().reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("teacher feature and score lengths differ")
    if target_kind == "mask":
        target = torch.zeros_like(y)
        target[record["teacher_keep_indices"].long()] = 1.0
        y = target
    else:
        y = (y - y.mean()) / (y.std(unbiased=False) + 1e-6)
    return x, y


def pairwise_ranking_loss(
    model: TokenScorePredictor,
    records: list[dict],
    device: str,
    max_records: int,
    margin: float,
) -> torch.Tensor:
    """Margin loss on teacher-high versus teacher-low token pairs."""
    if not records:
        return torch.zeros((), device=device)
    losses = []
    for record in records[:max_records]:
        x, y = target_for(record, "score")
        if x.shape[0] < 2:
            continue
        pred = model(x.to(device)).float()
        n = max(1, min(8, pred.numel() // 4))
        high = torch.topk(y, n).indices.to(device)
        low = torch.topk(-y, n).indices.to(device)
        losses.append(
            F.relu(margin - (pred[high][:, None] - pred[low][None, :])).mean()
        )
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def shard_tensors(records: list[dict], target_kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [target_for(record, target_kind) for record in records]
    if not pairs:
        raise ValueError("teacher shard contains no records")
    return torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs])


def evaluate(
    model: TokenScorePredictor,
    paths: list[Path],
    args: argparse.Namespace,
) -> tuple[float, float]:
    model.eval()
    mse_values, rank_values = [], []
    with torch.no_grad():
        for path in paths:
            _meta, records = load_shard(path)
            x, y = shard_tensors(records, args.target)
            for start in range(0, len(x), args.batch_size):
                xs = x[start : start + args.batch_size].to(args.device)
                ys = y[start : start + args.batch_size].to(args.device)
                if args.target == "mask":
                    value = F.binary_cross_entropy_with_logits(
                        model(xs).float(),
                        ys,
                        pos_weight=torch.tensor(3.0, device=args.device),
                    )
                else:
                    value = F.mse_loss(model(xs).float(), ys)
                mse_values.append(value.item())
            if args.ranking_weight:
                rank_values.append(
                    pairwise_ranking_loss(
                        model,
                        records,
                        args.device,
                        args.ranking_records,
                        args.ranking_margin,
                    ).item()
                )
    return (
        sum(mse_values) / max(1, len(mse_values)),
        sum(rank_values) / max(1, len(rank_values)),
    )


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    train_paths = shard_paths(args.teacher_path, "train")
    val_paths = shard_paths(args.val_path or args.teacher_path, "val")
    test_paths = (
        shard_paths(args.test_path or args.teacher_path, "test")
        if (args.test_path or Path(args.teacher_path).is_dir())
        else []
    )
    first_meta, first_records = load_shard(train_paths[0])
    input_dim = int(first_records[0]["predictor_features"].shape[-1])
    model = TokenScorePredictor(input_dim, args.hidden_dim, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_metric, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_mse, train_rank = [], []
        for path in train_paths:
            _meta, records = load_shard(path)
            x, y = shard_tensors(records, args.target)
            loader = DataLoader(
                TensorDataset(x, y),
                batch_size=args.batch_size,
                shuffle=True,
            )
            for xs, ys in loader:
                pred = model(xs.to(args.device)).float()
                if args.target == "mask":
                    mse = F.binary_cross_entropy_with_logits(
                        pred,
                        ys.to(args.device),
                        pos_weight=torch.tensor(3.0, device=args.device),
                    )
                else:
                    mse = F.mse_loss(pred, ys.to(args.device))
                rank = (
                    pairwise_ranking_loss(
                        model,
                        records,
                        args.device,
                        args.ranking_records,
                        args.ranking_margin,
                    )
                    if args.ranking_weight
                    else torch.zeros((), device=args.device)
                )
                loss = mse + args.ranking_weight * rank
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_mse.append(mse.item())
                train_rank.append(rank.item())
        val_mse, val_rank = evaluate(model, val_paths, args)
        metric = val_mse + args.ranking_weight * val_rank
        print(
            f"[PREDICTOR] epoch={epoch} "
            f"train_mse={sum(train_mse)/max(1,len(train_mse)):.6f} "
            f"train_rank={sum(train_rank)/max(1,len(train_rank)):.6f} "
            f"val_mse={val_mse:.6f} val_rank={val_rank:.6f}",
            flush=True,
        )
        if metric < best_metric:
            best_metric = metric
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    test_mse, test_rank = evaluate(model, test_paths, args) if test_paths else (None, None)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config_dict(),
            "meta": {
                "feature_source": first_meta.get("feature_source", "early_hidden"),
                "early_layer": int(first_meta.get("early_layer", 8)),
                "teacher_path": str(args.teacher_path),
                "target": args.target,
                "ranking_weight": args.ranking_weight,
                "val_mse": val_mse,
                "val_ranking_loss": val_rank,
                "test_mse": test_mse,
                "test_ranking_loss": test_rank,
            },
        },
        output,
    )
    print(
        f"[PREDICTOR] saved {output} val_mse={val_mse:.6f} test_mse={test_mse}",
        flush=True,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--teacher-path",
        required=True,
        help="Directory containing train/val/test manifests or one teacher file",
    )
    p.add_argument("--val-path", default=None)
    p.add_argument("--test-path", default=None)
    p.add_argument("--output-path", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--target", choices=["score", "mask"], default="score")
    p.add_argument("--ranking-weight", type=float, default=0.2)
    p.add_argument("--ranking-margin", type=float, default=0.1)
    p.add_argument("--ranking-records", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
