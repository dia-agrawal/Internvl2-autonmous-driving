#!/usr/bin/env python3
import os
import json
import argparse
import pathlib
import sys

import torch
from PIL import Image

# ----- make internvl_chat importable -----
repo_root = pathlib.Path(__file__).resolve().parents[1]
repo_root_str = str(repo_root)
internvl_chat_path = str(repo_root.joinpath("internvl_chat"))
for p in (internvl_chat_path, repo_root_str):
    if p not in sys.path:
        sys.path.insert(0, p)


def parse_args():
    p = argparse.ArgumentParser(description="Run InternVL on DriveLM val images and save JSONL.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to finetuned checkpoint folder (e.g. work_dirs/.../checkpoint-170080). "
             "If not set, uses --pretrained_model_path.",
    )
    p.add_argument(
        "--pretrained_model_path",
        type=str,
        default="pretrained/InternVL3-1B",
        help="Path to pretrained model (default: pretrained/InternVL3-1B).",
    )
    p.add_argument(
        "--img-root",
        required=True,
        help="Root folder of validation images (e.g. data/DriveLM/drivelm_nus_imgs_val/val_data)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output JSONL file to save predictions",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max number of images (0 = all)",
    )
    p.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
    )
    return p.parse_args()


def load_model_and_tokenizer(model_path, dtype="auto", device_map="auto"):
    """Same style as evaluate_bdd100k.py."""
    from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
    from transformers import AutoTokenizer
    import safetensors.torch  # noqa: F401

    # dtype selection
    if dtype == "auto":
        if torch.cuda.is_available():
            dtype_torch = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype_torch = torch.float32
    else:
        dtype_torch = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]

    print(f"[INFO] Loading model from {model_path} with dtype={dtype_torch}")
    config = InternVLChatConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    try:
        model = InternVLChatModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype_torch,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
    except Exception as e:
        print(f"[WARN] Primary load failed ({e}), falling back to device_map=None")
        model = InternVLChatModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype_torch,
            low_cpu_mem_usage=True,
            device_map=None,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

    model.eval()
    return model, tokenizer


def build_transform(input_size: int = 448):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def load_image_to_pixels(image_path: str, force_image_size: int = 448):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((force_image_size, force_image_size))
    transform = build_transform(input_size=force_image_size)
    pv = transform(img_resized)
    pixel_values = torch.stack([pv])  # [1, 3, H, W]
    return pixel_values


def generate_answer(model, tokenizer, pixel_values, prompt, max_new_tokens=128):
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device=device, dtype=next(model.parameters()).dtype)
    generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}

    with torch.no_grad():
        answer = model.chat(tokenizer, pixel_values, prompt, generation_config)
    return answer.strip()


def list_images(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


def main():
    args = parse_args()

    # choose model path
    if args.checkpoint:
        model_path = args.checkpoint
        print(f"[INFO] Using finetuned checkpoint: {model_path}")
    else:
        model_path = args.pretrained_model_path
        print(f"[INFO] Using pretrained model: {model_path}")

    model, tokenizer = load_model_and_tokenizer(model_path, dtype=args.dtype, device_map="auto")
    force_image_size = getattr(getattr(model, "config", None), "force_image_size", 448)

    img_root = os.path.expanduser(args.img_root)
    all_imgs = list_images(img_root)
    print(f"[INFO] Found {len(all_imgs)} images under {img_root}")

    max_n = args.max_samples if args.max_samples > 0 else len(all_imgs)
    all_imgs = all_imgs[:max_n]
    print(f"[INFO] Will run inference on {len(all_imgs)} images")

    question = "Describe this driving scene."
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_f = open(args.output, "w")

    for idx, img_path in enumerate(all_imgs, start=1):
        try:
            pixel_values = load_image_to_pixels(img_path, force_image_size=force_image_size)
            prompt = f"<image>\n{question}"
            prediction = generate_answer(model, tokenizer, pixel_values, prompt, max_new_tokens=128)
        except Exception as e:
            print(f"[WARN] Error on {img_path}: {e}")
            prediction = f"[Error: {str(e)[:100]}]"

        rel_path = os.path.relpath(img_path, img_root)

        ex = {
            "image": rel_path,        # relative to img_root for CLIP eval
            "question": question,
            "prediction": prediction,
        }
        out_f.write(json.dumps(ex) + "\n")

        if idx % 20 == 0:
            print(f"[INFO] Processed {idx}/{len(all_imgs)} images")

    out_f.close()
    print(f"[INFO] Done. Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
