"""
CelebV-HQ Dataset Pipeline Stage 1 - candidate selection / object category balancing.

Reads the full celebvhq_info.json (~35k clips) ONCE and writes a much smaller,
object category-balanced candidate.json that every later stage consumes:

    celebvhq_info.json  --(filter.py)-->  candidate.json
                                              |
                              acquire.py (stage2) downloads ONLY these clips
                              and carries `appearance` / `hair_color` forward.

Balancing (per project spec):
  - axis = hair COLOUR: 4 explicit colours + an "other" bucket = 5 buckets.
  - strategy = "fill the min per-bucket quota first, then let any bucket overflow
    to top the total up to --total". Rare colours (e.g. gray) simply contribute
    whatever supply exists; the deficit is back-filled from commoner buckets.
  - morphology (straight/wavy/long) is reported only, never balanced on.
  - multi-colour clips are assigned to their RAREST present colour (helps balance);
    clips with no colour go to "other"; hat/bald clips are excluded entirely.

candidate.json schema (drop-in for acquire.load_clips, + extra fields):
    {
      "meta_info": {"appearance_mapping": [...40...]},
      "clips": {
        "<clip_id>": {
          "ytb_id": "...", "duration": {...}, "bbox": {...},
          "appearance": [0/1 ...40...],     # raw vector (per-attribute tables later)
          "hair_color": "brown_hair"        # assigned balancing bucket
        }, ...
      }
    }

Usage:
    python -m data.celebv.pipeline.filter --total 10000 --seed 42
    python -m data.celebv.pipeline.filter --info data/celebv/pipeline/celebvhq_info.json \
        --out data/celebv/pipeline/candidate.json --total 10000
"""


from __future__ import annotations
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from data.celebv.pipeline.attributes import (
    APPEARANCE_MAPPING,
    ACTION_MAPPING,
    HAIR_BUCKETS,
    OTHER_BUCKET,
    colors_present,
    morphology_present,
    is_excluded,
    IDX,
)


# ============================================================
#                       info.json loading
# ============================================================
def load_info(info_path: Path) -> Dict[str, Any]:
    if not info_path.exists():
        raise FileNotFoundError(f"celebvhq_info.json not found: {info_path}")
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    mapping = info.get("meta_info", {}).get("appearance_mapping")
    if mapping is not None and list(mapping) != APPEARANCE_MAPPING:
        # Hard fail: bucket + caption logic is index-based, a reordered mapping
        # would silently corrupt every label. Update attributes.APPEARANCE_MAPPING
        # if CelebV-HQ ever ships a new schema.
        raise ValueError(
            "celebvhq_info.json appearance_mapping does not match "
            "attributes.APPEARANCE_MAPPING; refusing to proceed (index-based decode "
            "would be wrong). Sync the constant first."
        )
    act_mapping = info.get("meta_info", {}).get("action_mapping")
    if act_mapping is not None and list(act_mapping) != ACTION_MAPPING:
        raise ValueError(
            "celebvhq_info.json action_mapping does not match "
            "attributes.ACTION_MAPPING; refusing to proceed (index-based decode "
            "would be wrong). Sync the constant first."
        )
    if "clips" not in info:
        raise ValueError("celebvhq_info.json has no top-level 'clips' object.")
    return info


# def _appearance_of(clip_val: Dict[str, Any]) -> List[int]:
#     return list(clip_val["attributes"]["appearance"])


# ============================================================
#                    Bucketing + balancing
# ============================================================
def assign_bucket(appearance: List[int], color_freq: Counter) -> str:
    """Single colour -> that colour; many -> rarest present (balances); none -> other."""
    present = colors_present(appearance)
    if len(present) == 1:
        return present[0]
    if len(present) == 0:
        return OTHER_BUCKET
    return min(present, key=lambda c: color_freq[c]) # return the rarest color bucket
    # min(present, key=lambda c: color_freq[c]) -> = min{ def f(c): return color_freq[c] (c in present) }
    # where color_freq is a Counter object


def select_balanced(
    info: Dict[str, Any], total: int, seed: int,
) -> Tuple[List[Tuple[str, Dict[str, Any], List[int], str]], Dict[str, Any]]:
    """Return (selected items, stats). Each item = (clip_id, clip_val, appearance, bucket)."""
    clips: Dict[str, Any] = info["clips"]
    rng = random.Random(seed)

    # ---- pass 1: gather eligible clips + global colour frequency statistics ----
    eligible: List[Tuple[str, Dict[str, Any], List[int]]] = []
    color_freq: Counter = Counter()
    n_excluded = 0
    for cid, val in clips.items():
        try:
            app = list(val["attributes"]["appearance"])
        except (KeyError, TypeError):
            continue
        if is_excluded(app):
            n_excluded += 1
            continue
        eligible.append((cid, val, app))
        for c in colors_present(app):
            color_freq[c] += 1

    # ---- pass 2: assign each eligible clip to one bucket ----
    buckets: Dict[str, List[Tuple[str, Dict[str, Any], List[int], str]]] = {
        b: [] for b in HAIR_BUCKETS
    }
    for cid, val, app in eligible:
        b = assign_bucket(app, color_freq)
        buckets[b].append((cid, val, app, b))

    avail = {b: len(buckets[b]) for b in HAIR_BUCKETS}
    for b in HAIR_BUCKETS: # shuffle the clips in each bucket
        rng.shuffle(buckets[b])

    # ---- pass 3: quota fill + overflow back-fill ----
    # theory: fill each bucket to the quota first, then let any bucket overflow
    per = total // len(HAIR_BUCKETS)
    selected: List[Tuple[str, Dict[str, Any], List[int], str]] = []
    leftovers: List[Tuple[str, Dict[str, Any], List[int], str]] = []
    for b in HAIR_BUCKETS:
        selected.extend(buckets[b][:per])
        leftovers.extend(buckets[b][per:])
    if len(selected) < total and leftovers:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: total - len(selected)])

    stats = {
        "total_clips_in_info": len(clips),
        "excluded_hat_or_bald": n_excluded,
        "eligible": len(eligible),
        "color_freq_global": dict(color_freq),
        "available_per_bucket": avail,
        "per_bucket_quota": per,
        "requested_total": total,
        "selected_total": len(selected),
    }
    return selected, stats


# ============================================================
#                    Output + reporting
# ============================================================
def _action_of(val: Dict[str, Any]) -> List[int]:
    """Raw 35-d action vector (empty if missing)."""
    try:
        return list(val["attributes"]["action"])
    except (KeyError, TypeError):
        return []


def build_candidate(selected) -> Dict[str, Any]:
    out_clips: Dict[str, Any] = {}
    for cid, val, app, bucket in selected:
        out_clips[cid] = {
            "ytb_id":     val["ytb_id"],
            "duration":   val["duration"],
            "bbox":       val["bbox"],
            "appearance": app,
            "action":     _action_of(val),   # descriptive only; appended to caption tail
            "hair_color": bucket,
        }
    return {
        "meta_info": {
            "appearance_mapping": APPEARANCE_MAPPING,
            "action_mapping": ACTION_MAPPING,
        },
        "clips": out_clips,
    }


def report(selected, stats) -> None:
    sel_bucket = Counter(b for _, _, _, b in selected)
    morph = Counter()
    gender = Counter()
    for _, _, app, _ in selected:
        for m in morphology_present(app):
            morph[m] += 1
        gender["male" if (IDX["male"] < len(app) and app[IDX["male"]]) else "female"] += 1

    print("\n========== filter.py distribution report ==========")
    print(f"info clips           : {stats['total_clips_in_info']}")
    print(f"excluded (hat/bald)  : {stats['excluded_hat_or_bald']}")
    print(f"eligible             : {stats['eligible']}")
    print(f"requested total      : {stats['requested_total']}")
    print(f"selected total       : {stats['selected_total']}  (per-bucket quota={stats['per_bucket_quota']})")
    print("\n-- hair-color buckets (selection axis) --")
    for b in HAIR_BUCKETS:
        print(f"  {b:<12} selected={sel_bucket.get(b, 0):<6} available={stats['available_per_bucket'].get(b, 0)}")
    print("\n-- hair morphology (descriptive only) --")
    for m, c in morph.most_common():
        print(f"  {m:<14} {c}")
    print("\n-- gender --")
    for g, c in gender.most_common():
        print(f"  {g:<8} {c}")
    print("===================================================\n")


# ============================================================
#                            CLI
# ============================================================
def main():
    p = argparse.ArgumentParser("CelebV-HQ stage1 filter -> balanced candidate.json")
    p.add_argument("--info", default="data/celebv/pipeline/celebvhq_info.json",
                   help="full CelebV-HQ info json")
    p.add_argument("--out", default="data/celebv/pipeline/candidate.json",
                   help="output candidate manifest (consumed by acquire.py --info)")
    p.add_argument("--total", type=int, default=10000,
                   help="number of clips to select into candidate.json")
    p.add_argument("--seed", type=int, default=42, help="deterministic selection seed")
    p.add_argument("--force", action="store_true", help="overwrite existing candidate.json")
    args = p.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise SystemExit(f"refuse to overwrite existing {out_path} (pass --force)")

    info = load_info(Path(args.info))
    selected, stats = select_balanced(info, total=args.total, seed=args.seed)
    if not selected:
        raise SystemExit("[filter] no eligible clips selected; check info json / exclusions")

    candidate = build_candidate(selected)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(candidate, f, ensure_ascii=False)
    import os
    os.replace(tmp, out_path)

    report(selected, stats)
    print(f"[filter] wrote {len(candidate['clips'])} clips -> {out_path}")


if __name__ == "__main__":
    main()
