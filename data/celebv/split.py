"""
CelebV train/val splitter.

src:
    {prepare.out_root}/index.jsonl     # produced by pipeline/main.py
    each line = {"clip_id", "ytb_id", "n_refs", "path"}
out:
    atomic write -> {split_dir}/{train,val}.jsonl

Usage:
    python -m data.celebv.split                        # default 0.1 val
    python -m data.celebv.split --val-ratio 0.05
    python -m data.celebv.split --seed 7 --force
"""

from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from data.utils import load_cfg


# ============================================================
#                       I/O helpers
# ============================================================
def read_index(path: Path) -> List[Dict[str, Any]]:
    """Read index.jsonl, drop blank lines, dedupe by clip_id (last write wins)."""
    if not path.exists():
        raise FileNotFoundError(f"index.jsonl not found: {path}; run pipeline/main.py first")
    seen: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            cid = row.get("clip_id")
            if not cid:
                continue
            seen[cid] = row
    return list(seen.values())


def write_jsonl_atomic(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write to .tmp then os.replace -> readers never see a half-written split."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ============================================================
#                       Split logic
# ============================================================
def split_random(
    rows: List[Dict[str, Any]], val_ratio: float, rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Pure random shuffle + cut at floor(N * val_ratio)."""
    pool = list(rows)
    rng.shuffle(pool)
    n_val = max(1, int(len(pool) * val_ratio)) if pool else 0
    return pool[n_val:], pool[:n_val]


# ============================================================
#                            CLI
# ============================================================
def main():
    p = argparse.ArgumentParser("split celebv index.jsonl into train/val.jsonl")
    p.add_argument("--config", default="data/celebv/config.yaml",
                   help="dataset cfg; reads prepare.out_root / prepare.index_file / split_dir")
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help="fraction of clips for val (default 0.1)")
    p.add_argument("--seed", type=int, default=42, help="deterministic shuffle seed")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing {train,val}.jsonl")
    p.add_argument("--dry-run", action="store_true", help="print counts only, do not write")
    args = p.parse_args()

    if not (0.0 < args.val_ratio < 1.0):
        raise SystemExit(f"--val-ratio must be in (0, 1), got {args.val_ratio}")

    cfg = load_cfg(args.config)
    index_path = Path(cfg.prepare.out_root) / cfg.prepare.index_file
    split_dir = Path(cfg.split_dir)
    train_path = split_dir / "train.jsonl"
    val_path = split_dir / "val.jsonl"

    if (train_path.exists() or val_path.exists()) and not args.force and not args.dry_run:
        raise SystemExit(
            f"refuse to overwrite existing splits under {split_dir} (pass --force)"
        )

    print(f"[split] reading {index_path}")
    rows = read_index(index_path)
    print(f"[split] index size = {len(rows)}")
    if not rows:
        raise SystemExit("[split] empty index; nothing to split")

    rng = random.Random(args.seed)
    train, val = split_random(rows, args.val_ratio, rng)

    print(f"[split] mode=random  seed={args.seed}  val_ratio={args.val_ratio}")
    print(f"[split] train={len(train)}  val={len(val)}")

    if args.dry_run:
        print("[split] dry-run; no files written")
        return

    write_jsonl_atomic(train, train_path)
    write_jsonl_atomic(val,   val_path)
    print(f"[split] wrote {train_path}")
    print(f"[split] wrote {val_path}")


if __name__ == "__main__":
    main()
