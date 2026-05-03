import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TIMM_USE_OLD_CACHE"] = "1"
# 这部分必须放在最前面 防止pyiqa从huggingface下载权重

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import pyiqa
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def safe_torch_load(path: str):
    """
    Compatible with different PyTorch versions.  
    """
    # FIXME: 这里默认直接使用 2.x 版本的pytorch了 不需要再分辨
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def infer_weight_keys(weight_path: str, mode: str = "auto") -> Optional[str]:
    """
    PyIQA train 出来的权重通常是:
        {"params": state_dict, ...}

    原始 MANIQA 官方 release 权重通常直接就是 state_dict
    这种情况下需要 weight_keys=None。
    """
    # FIXME： 已经明确直接使用存放在weights文件夹下的权重文件 与pyiqa的训练无关 所以不需要再进行mode的推断了
    if mode == "none":
        return None

    if mode != "auto":
        return mode

    ckpt = safe_torch_load(weight_path)

    if isinstance(ckpt, dict):
        for key in ["params", "state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return key

    return None


def collect_images(input_path: str):
    p = Path(input_path)

    if p.is_file():
        return [p]

    if p.is_dir():
        image_paths = []
        for x in p.rglob("*"):
            if x.suffix.lower() in IMAGE_EXTS:
                image_paths.append(x)
        return sorted(image_paths)

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def create_maniqa_metric(
    metric_name: str,
    device: torch.device,
    weight_path: Optional[str] = None,
    weight_keys_mode: str = "auto",
    test_sample: int = 20,
):
    """
    metric_name:
        maniqa        -> KONIQ-10K
        maniqa-kadid  -> KADID-10K
        maniqa-pipal  -> PIPAL
    """

    custom_opts = {
        "test_sample": test_sample,
    }

    # 如果你要手动加载本地权重，就不要让 pyiqa 先加载默认权重
    if weight_path is not None:
        custom_opts["pretrained"] = False

    metric = pyiqa.create_metric(
        metric_name,
        device=device,
        **custom_opts,
    )

    loaded_weight_keys = None

    if weight_path is not None:
        loaded_weight_keys = infer_weight_keys(weight_path, weight_keys_mode)

        print(f"[INFO] Loading local weights: {weight_path}")
        print(f"[INFO] weight_keys = {loaded_weight_keys}")

        metric.load_weights(
            weight_path,
            weight_keys=loaded_weight_keys,
        )

    metric.eval()
    return metric, loaded_weight_keys


def score_one_image(metric, image_path: Path) -> float:
    with torch.inference_mode():
        score = metric(str(image_path))

    if isinstance(score, torch.Tensor):
        return float(score.detach().cpu().flatten()[0].item())

    return float(score)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="./test.jpg",
        help="Path to image file or image folder.",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="./weights/MANIQA/maniqa.pt",
        help="Local MANIQA weight path. Set empty string to use pyiqa default weights.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="maniqa",
        choices=["maniqa", "maniqa-kadid", "maniqa-pipal"],
        help="MANIQA variant in pyiqa.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda, cuda:0, or cpu. Default: auto.",
    )

    parser.add_argument(
        "--weight_keys",
        type=str,
        default="auto",
        help="auto, none, params, state_dict, model, net.",
    )

    parser.add_argument(
        "--test_sample",
        type=int,
        default=20,
        help="Number of 224x224 crops used inside MANIQA inference.",
    )

    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Optional output jsonl path.",
    )

    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    weight_path = args.weights.strip()
    if weight_path == "":
        weight_path = None

    image_paths = collect_images(args.input)

    metric, loaded_weight_keys = create_maniqa_metric(
        metric_name=args.metric,
        device=device,
        weight_path=weight_path,
        weight_keys_mode=args.weight_keys,
        test_sample=args.test_sample,
    )

    results = []

    fout = None
    if args.output_jsonl is not None:
        fout = open(args.output_jsonl, "w", encoding="utf-8")

    for image_path in tqdm(image_paths, desc="MANIQA inference"):
        record = {
            "image_path": str(image_path),
            "metric": args.metric,
            "weights": weight_path,
            "weight_keys": loaded_weight_keys,
            "score": None,
            "lower_better": bool(getattr(metric, "lower_better", False)),
            "score_range": getattr(metric, "score_range", None),
            "error": None,
        }

        try:
            score = score_one_image(metric, image_path)
            record["score"] = score
        except Exception as e:
            record["error"] = repr(e)

        results.append(record)

        if fout is not None:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    if fout is not None:
        fout.close()

    if len(results) == 1:
        print(json.dumps(results[0], ensure_ascii=False, indent=2))
    else:
        valid_scores = [x["score"] for x in results if x["score"] is not None]
        avg_score = sum(valid_scores) / max(len(valid_scores), 1)

        print(json.dumps({
            "num_images": len(results),
            "num_valid": len(valid_scores),
            "metric": args.metric,
            "weights": weight_path,
            "avg_score": avg_score,
            "output_jsonl": args.output_jsonl,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()