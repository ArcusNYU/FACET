"""
Dataset Pipeline Stage 3: Latent cache (T5 caption embedding + Wan VAE video latent).
Reference: https://github.com/Wan-Video/Wan2.2

Reads {out_root}/index.jsonl produced by main.py. For each qualified clip:
  1. Load {clip_dir}/meta.json -> caption.
  2. Load {clip_dir}/{cid}.mp4 -> first cfg.num_frames frames as [3,T,H,W] in [-1,1].
  3. Encode caption with T5EncoderModel -> [L, 4096] (variable length, padded by DiT at runtime).
  4. Encode video with Wan2_2_VAE -> [48, T', H', W']  (already mean-std normalized).
  5. Save {tgt_latent, t5_emb} to {latent_cache_dir}/{part}/{ab}/{cd}/{cid}.pt
     atomically (.pt.tmp -> rename), so a killed run never leaves a corrupt file.

Resume: at startup latent_root is “rglob”ed once to build a set of already-cached
        clip_ids; matching items are dropped from todo. Stale `.pt.tmp` files
        from killed runs are also swept. 
        With `--shard i/N` each worker only handles items whose hash(cid) % N == i
Disk dtype: defaults to bfloat16 to keep total cache small
            (~2.6 MB / clip for 480x832x81 @ z_dim=48). Override with --save-dtype.

Wan weights layout (cfg.prepare.weight_dir / WAN):
    models_t5_umt5-xxl-enc-bf16.pth          # T5 encoder ckpt
    google/umt5-xxl/                         # T5 tokenizer (HF format)
    Wan2.2_VAE.pth                           # video VAE ckpt
"""

# TODO: 单进程串行版本; 大数据集时可用 CUDA_VISIBLE_DEVICES + --shard 切片并发
# TODO: 若后续允许 negative-prompt CFG dropout, 在此一并 cache null T5 embedding


from __future__ import annotations
import argparse
import hashlib
import json
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")  # A100

import warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*torch\.cuda\.amp\.autocast.*")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*weights_only=False.*")

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm

_WAN_PATH = Path(__file__).resolve().parents[3] / "WAN"
if str(_WAN_PATH) not in sys.path:
    sys.path.insert(0, str(_WAN_PATH))

from wan.modules.t5 import T5EncoderModel       # noqa: E402
from wan.modules.vae2_2 import Wan2_2_VAE       # noqa: E402

from data.utils import load_cfg                 # noqa: E402


# ============================================================
#                       Index / paths
# ============================================================
def read_index(path: Path) -> List[Dict[str, Any]]:
    """Read index.jsonl; dedupe by clip_id."""
    if not path.exists():
        raise FileNotFoundError(f"index.jsonl missing: {path}; run main.py first")
    seen: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "clip_id" in obj and "part" in obj:
                seen[obj["clip_id"]] = obj
    return list(seen.values())


def clip_dir(out_root: Path, part: str, cid: str) -> Path:
    return out_root / "clips" / part / cid[:2] / cid[2:4] / cid


def latent_pt_path(latent_root: Path, part: str, cid: str) -> Path:
    """latent_root/{part}/{ab}/{cd}/{cid}.pt   (matches openvid.py _load_cache)."""
    return latent_root / part / cid[:2] / cid[2:4] / f"{cid}.pt"


def scan_cached_cids(latent_root: Path) -> set[str]:
    """One-shot directory walk: collect every {cid}.pt already on disk.
    """
    if not latent_root.exists():
        return set()
    return {p.stem for p in latent_root.rglob("*.pt")}


def sweep_stale_tmp(latent_root: Path) -> int:
    """Remove `.pt.tmp` left behind by a killed run."""
    if not latent_root.exists():
        return 0
    n = 0
    for p in latent_root.rglob("*.pt.tmp"):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


# ============================================================
#                    Video read & normalize
# ============================================================
def read_video_tensor(mp4: Path, num_frames: int) -> torch.Tensor:
    """Read first `num_frames` frames -> [3,T,H,W] float32 in [-1,1].
       Pads by repeating the last frame if the source is shorter (defensive;
       main.py already filters short clips, thus a no-op func in practice).
    """
    # NOTE: VAE requires float32 video tensor
    vr = VideoReader(str(mp4), ctx=cpu(0))
    n = len(vr)
    if n < num_frames:
        idx = list(range(n)) + [n - 1] * (num_frames - n)
    else:
        idx = list(range(num_frames))
    frames = vr.get_batch(idx).asnumpy()                    # [T,H,W,3] uint8 (RGB)
    t = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous().float() # [3,T,H,W]
    t = t.div_(127.5).sub_(1.0)                             # [-1, 1]
    return t                                                # [3,T,H,W]


# ============================================================
#                       Atomic .pt save
# ============================================================
def save_atomic(payload: Dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


# ============================================================
#                          Encoders
# ============================================================
def encode_caption(text_encoder: T5EncoderModel,
                   caption: str,
                   device: torch.device) -> torch.Tensor:
    """Returns [L, 4096] in T5 dtype (bf16). pad-trim is done by DiT at runtime."""
    if not caption or not caption.strip():
        caption = " "
    with torch.no_grad():
        emb_list = text_encoder([caption], device)          # List[Tensor [L_i, 4096]]
    return emb_list[0].detach()


def encode_video(vae: Wan2_2_VAE,
                 video: torch.Tensor,
                 device: torch.device) -> torch.Tensor:
    """video: [3,T,H,W] in [-1,1]. Returns [48, T', H', W'] float32 (already
       mean-std normalized inside Wan2_2_VAE.encode via self.scale)."""
    with torch.no_grad():
        z_list = vae.encode([video.to(device, non_blocking=True)])
    return z_list[0].detach()


# ============================================================
#                            Main
# ============================================================
def main():
    p = argparse.ArgumentParser("Cache T5 caption embeddings + Wan VAE video latents")
    p.add_argument("--config", default=str(Path(__file__).parent.parent / "config.yaml"))
    p.add_argument("--limit", type=int, default=-1,
                   help="optional cap on total clips to cache (debug); -1 = all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="storage dtype for tgt_latent & t5_emb (default bf16, halves disk)")
    p.add_argument("--shard", default="0/1",
                   help="i/N: this worker handles items whose hash(cid)%%N == i. "
                        "Use to spread 100k+ clips across multiple GPUs/processes.")
    p.add_argument("--no-sweep-tmp", action="store_true",
                   help="skip cleanup of stale .pt.tmp files at startup")
    # 当前阶段不使用该指令 默认清除未完成临时文件
    args = p.parse_args()

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    assert 0 <= shard_i < shard_n, f"bad --shard {args.shard}"

    cfg = load_cfg(args.config)
    out_root = Path(cfg.prepare.out_root)
    weight_dir = Path(cfg.prepare.weight_dir)
    latent_root = Path(cfg.latent_cache_dir)
    index_path = out_root / cfg.prepare.index_file
    NF = int(cfg.num_frames)

    save_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.save_dtype]

    # ---- index & resume ----
    items = read_index(index_path)
    if args.limit > 0:
        items = items[: args.limit]

    if not args.no_sweep_tmp:
        n_tmp = sweep_stale_tmp(latent_root)
        if n_tmp:
            print(f"[cache] swept {n_tmp} stale .pt.tmp files")

    # Build the "already cached" set in ONE directory walk (cheap on 100k items)
    cached_cids = scan_cached_cids(latent_root)

    todo: List[Dict[str, Any]] = []
    n_skipped_shard = 0
    for it in items:
        cid = it["clip_id"]
        if cid in cached_cids:
            continue
        # Stable cross-process hash (built-in hash() is seeded per-process).
        if shard_n > 1 and (int(hashlib.md5(cid.encode()).hexdigest(), 16) % shard_n) != shard_i:
            n_skipped_shard += 1
            continue
        todo.append(it)

    n_cached = len(cached_cids & {it["clip_id"] for it in items})
    print(f"[cache] indexed={len(items)}  cached={n_cached}  "
          f"shard={args.shard} (other_shards={n_skipped_shard})  todo={len(todo)}")
    if not todo:
        return

    device = torch.device(args.device)

    # ---- load Wan models ----
    wan_root = weight_dir / "WAN"
    t5_ckpt = wan_root / "models_t5_umt5-xxl-enc-bf16.pth"
    t5_tok = wan_root / "google" / "umt5-xxl"
    vae_ckpt = wan_root / "Wan2.2_VAE.pth"

    print(f"[cache] loading T5  from {t5_ckpt}")
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=str(t5_ckpt),
        tokenizer_path=str(t5_tok),
    )
    print(f"[cache] loading VAE from {vae_ckpt}")
    vae = Wan2_2_VAE(vae_pth=str(vae_ckpt), device=str(device))

    # ---- per-clip cache ----
    n_ok, n_fail = 0, 0
    for it in tqdm(todo, desc="cache"):
        cid = it["clip_id"]
        part = it["part"]
        cdir = clip_dir(out_root, part, cid)
        mp4 = cdir / f"{cid}.mp4"
        meta_path = cdir / "meta.json"

        if not mp4.exists() or not meta_path.exists():
            print(f"\n[cache] skip {cid}: missing mp4 or meta.json", flush=True)
            n_fail += 1
            continue

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            caption = str(meta.get("caption", "")).strip()

            t5_emb = encode_caption(text_encoder, caption, device)        # [L, 4096] bf16
            video = read_video_tensor(mp4, NF)                            # [3,T,H,W] fp32 [-1,1]
            tgt_latent = encode_video(vae, video, device)                 # [48,T',H',W'] fp32

            payload = {
                "tgt_latent": tgt_latent.to(save_dtype).cpu().contiguous(),
                "t5_emb":     t5_emb.to(save_dtype).cpu().contiguous(),
            }
            save_atomic(payload, latent_pt_path(latent_root, part, cid))
            n_ok += 1
        except Exception as e:
            print(f"\n[cache] FAIL {cid}: {type(e).__name__}: {e}", flush=True)
            n_fail += 1
        finally:
            torch.cuda.empty_cache()

    print(f"[cache] done. ok={n_ok}  fail={n_fail}")


if __name__ == "__main__":
    main()
