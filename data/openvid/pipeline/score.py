"""
Dataset Pipeline Step 3 (per-clip): ref-frame candidate scoring.
Reference: https://github.com/chaofengc/IQA-PyTorch, https://github.com/iigroup/maniqa, https://github.com/QwenLM/Qwen3-VL

3 Layers workflow: applied in order per random-sampled candidate frame.
L1 cv_check  : pure-rule check on bbox side + mask coverage ratio. zero model.
L2 IqaScorer : pyiqa + MANIQA (loaded from local weights/MANIQA/maniqa.pt).  
               HF env vars are forced OFFLINE before pyiqa import so it doesn't 
               try to fetch the upstream MANIQA weight at runtime.
L3 VlmFilter : Qwen3-VL-8B-Instruct judge. Prompt template loaded once from
               data/openvid/pipeline/prompt.txt; `{category}` is interpolated
               per-call so the same VLM can serve multiple target categories.

All three components are stateful classes; instantiate once per process.
"""

from __future__ import annotations
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TIMM_USE_OLD_CACHE", "1")

import json
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image


# ---- L1: zero-cost rule check ----
def cv_check(
    bbox_hw: Tuple[int, int],
    mask_ratio: float,
    category: str,
    cv_min_size: Dict[str, int],
    cv_mask_ratio: Dict[str, list],
) -> bool:
    """Pre-filter on (bbox short side, mask-pixels / bbox-pixels).
    Returns True if the candidate frame is acceptable for this category.
    """
    if category not in cv_min_size or category not in cv_mask_ratio:
        # unknown category -> conservative pass (let later layers decide)
        return True
    h, w = bbox_hw
    if min(h, w) < int(cv_min_size[category]):
        return False
    lo, hi = cv_mask_ratio[category]
    return float(lo) <= float(mask_ratio) <= float(hi)


# ---- L2: IQA via pyiqa MANIQA ----
class IqaScorer:
    """pyiqa + MANIQA wrapper. score(rgb_uint8) -> float."""

    def __init__(
        self,
        weight_path: str | Path,
        metric: str = "maniqa",
        device: str = "cuda",
        test_sample: int = 20,
    ):
        import torch
        import pyiqa

        self.device = torch.device(device)
        self.metric = pyiqa.create_metric(
            metric, device=self.device, test_sample=test_sample, pretrained=False,
        )
        # Original MANIQA release weights are a flat state_dict (no pyiqa wrapper) -> weight_keys=None.
        self.metric.load_weights(str(weight_path), weight_keys=None)
        self.metric.eval()

    def score(self, img: np.ndarray | str | Path) -> float:
        """Accept either an [H,W,3] uint8 RGB array or a path-like to an image file."""
        import torch
        if isinstance(img, np.ndarray):
            # [H,W,3] uint8 -> [1,3,H,W] float in [0,1]
            t = (torch.from_numpy(img)
                    .float().permute(2, 0, 1).unsqueeze(0).div_(255.0)  #TODO: ?????
                    .to(self.device))
            with torch.inference_mode():
                s = self.metric(t)
        else:
            with torch.inference_mode():
                s = self.metric(str(img))

        if hasattr(s, "detach"):
            return float(s.detach().cpu().flatten()[0].item())
        return float(s)


# ---- L3: VLM judge via Qwen3-VL-8B-Instruct ----
def _extract_json(text: str) -> Dict[str, bool]:
    """Tolerant JSON extractor for VLM output. Defaults to a 'reject' triple on parse failure."""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {"match": False, "occlusion": True, "truncation": True}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"match": False, "occlusion": True, "truncation": True}
    return {
        "match":      bool(d.get("match", False)),
        "occlusion":  bool(d.get("occlusion", True)),
        "truncation": bool(d.get("truncation", True)),
    }


class VlmFilter:
    """Qwen3-VL-8B-Instruct ref filter.

    Interface:
        vlm = VlmFilter(model_dir, prompt_file)
        d = vlm.judge(rgb_uint8, category="upper_clothes")
        accept = d["match"] and not d["occlusion"] and not d["truncation"]
    """

    def __init__(
        self,
        model_dir: str | Path,
        prompt_file: str | Path,
        device_map: str = "auto",
        max_new_tokens: int = 96,
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.prompt_template = Path(prompt_file).read_text(encoding="utf-8")
        self.max_new_tokens = max_new_tokens
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir), dtype="auto", device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(str(model_dir))

    def _build_prompt(self, category: str) -> str:
        return self.prompt_template.format(category=category)

    def judge(self, img: np.ndarray | Image.Image | str | Path, category: str) -> Dict[str, bool]:
        import torch
        if isinstance(img, np.ndarray):
            pil = Image.fromarray(img).convert("RGB")
        elif isinstance(img, Image.Image):
            pil = img.convert("RGB")
        else:
            pil = Image.open(img).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil},        # PIL object directly (qwen.py learned that lesson)
                {"type": "text",  "text": self._build_prompt(category)},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        trimmed = [out_ids[len(in_ids):]
                   for in_ids, out_ids in zip(inputs.input_ids, generated)]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        return _extract_json(text)
