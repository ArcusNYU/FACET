"""
CelebV train/val splitter.

src:
    {prepare.out_root}/index.jsonl       # produced by pipeline/main.py
        each line = {"clip_id", "ytb_id", "n_refs", "path", "hair_color"}
    {prepare.raw_video_root}/downloaded.json
out:
    atomic write -> {split_dir}/{train,val}.jsonl

Leakage safety:
    Clips are grouped by `ytb_id` and WHOLE groups are assigned to train OR val,
    never split across both. The same YouTube source (hence the same identity /
    scene) therefore cannot appear on both sides -> no identity leakage, which is
    the standard "group split" required for a credible test protocol.

Usage:
    python -m data.celebv.split                        # default 0.1 val, group-by-ytb
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


def attach_attrs(rows: List[Dict[str, Any]], downloaded_path: Path) -> int:
    """In-place join `appearance` / `hair_color` from downloaded.json onto each row."""
    if not downloaded_path.exists():
        print(f"[split] downloaded.json not found at {downloaded_path}; "
              f"skipping attribute join (rows keep index.jsonl fields only)")
        return 0
    with open(downloaded_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = 0
    for r in rows:
        v = data.get(r.get("clip_id"), {}) or {}
        app = v.get("appearance")
        if app:
            r["appearance"] = list(app)
            n += 1
        if v.get("hair_color") and not r.get("hair_color"):
            r["hair_color"] = v["hair_color"]
    return n


# ============================================================
#                       Split logic
# ============================================================
def split_random(
    rows: List[Dict[str, Any]], val_ratio: float, rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Pure random shuffle + cut at floor(N * val_ratio). (No leakage guard.)"""
    pool = list(rows)
    rng.shuffle(pool)
    n_val = max(1, int(len(pool) * val_ratio)) if pool else 0
    return pool[n_val:], pool[:n_val]


def split_grouped_by_ytb(
    rows: List[Dict[str, Any]], val_ratio: float, rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Leakage-safe split: shuffle ytb_id GROUPS, assign whole groups to val until
    the val clip count reaches ceil(N * val_ratio), the rest to train. Guarantees
    no ytb_id (identity/scene) appears in both splits.

    Clips lacking a ytb_id fall back to their own clip_id as a singleton group."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    # group clips by ytb_id:
    for r in rows:
        key = r.get("ytb_id") or r.get("clip_id")
        groups.setdefault(key, []).append(r) # setdefault: if key not in groups, set groups[key] = []

    keys = list(groups.keys()) # shuffle the groups
    rng.shuffle(keys)

    # calculate the target number of val clips:
    n_val_target = max(1, int(len(rows) * val_ratio)) if rows else 0
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for k in keys:
        g = groups[k]
        if len(val) < n_val_target:
            val.extend(g)
        else:
            train.extend(g)
    return train, val


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
    p.add_argument("--no-group", action="store_true",
                   help="use a plain random split instead of the leakage-safe "
                        "group-by-ytb split (NOT recommended; risks identity leakage)")
    args = p.parse_args()

    if not (0.0 < args.val_ratio < 1.0):
        raise SystemExit(f"--val-ratio must be in (0, 1), got {args.val_ratio}")

    cfg = load_cfg(args.config)
    index_path = Path(cfg.prepare.out_root) / cfg.prepare.index_file
    downloaded_path = Path(cfg.prepare.raw_video_root) / "downloaded.json"
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

    n_attr = attach_attrs(rows, downloaded_path)
    print(f"[split] attached appearance attributes to {n_attr}/{len(rows)} rows "
          f"from {downloaded_path.name}")

    rng = random.Random(args.seed)
    if args.no_group:
        train, val = split_random(rows, args.val_ratio, rng)
        mode = "random"
    else:
        train, val = split_grouped_by_ytb(rows, args.val_ratio, rng)
        mode = "group-by-ytb"

    n_ytb_train = len({r.get("ytb_id") for r in train})
    n_ytb_val = len({r.get("ytb_id") for r in val})
    leaked = {r.get("ytb_id") for r in train} & {r.get("ytb_id") for r in val}
    print(f"[split] mode={mode}  seed={args.seed}  val_ratio={args.val_ratio}")
    print(f"[split] train={len(train)} ({n_ytb_train} ytb)  val={len(val)} ({n_ytb_val} ytb)")
    print(f"[split] ytb_id overlap (must be 0 for group mode): {len(leaked)}")

    if args.dry_run:
        print("[split] dry-run; no files written")
        return

    write_jsonl_atomic(train, train_path)
    write_jsonl_atomic(val,   val_path)
    print(f"[split] wrote {train_path}")
    print(f"[split] wrote {val_path}")


if __name__ == "__main__":
    main()
