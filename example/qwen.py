import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def build_prompt(category: str) -> str:
    return f"""
Please describe this image.
""".strip()


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in output: {text}")

    data = json.loads(match.group(0))
    result = {
        "category_match": bool(data.get("category_match", False)),
        "occlusion": bool(data.get("occlusion", True)),
        "truncation": bool(data.get("truncation", True)),
    }
    return result


def run_inference(model_path: str, image_path: str, category: str) -> dict:
    # ⭐ 关键改动：用 PIL 打开图片，转成 RGB，绕开 torchvision 格式限制
    image = Image.open(image_path).convert("RGB")
    prompt = build_prompt(category)

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(model_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,        # ⭐ 传 PIL 对象，不传 URI
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return extract_json(output_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./weights/Qwen8B")
    parser.add_argument("--image", type=str, default="./test.png")
    parser.add_argument("--category", type=str, default="upper_clothes")
    args = parser.parse_args()

    result = run_inference(
        model_path=args.model,
        image_path=args.image,
        category=args.category,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()