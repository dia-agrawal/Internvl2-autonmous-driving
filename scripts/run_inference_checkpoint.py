#!/usr/bin/env python3
"""Run inference against a saved checkpoint folder.

Usage examples:
  python scripts/run_inference_checkpoint.py \
    --checkpoint work_dirs/.../checkpoint-170080 \
    --prompt "Hello, who are you?"

  python scripts/run_inference_checkpoint.py \
    --checkpoint work_dirs/.../checkpoint-170080 \
    --image data/path/to.jpg \
    --prompt "<image>\nPlease describe the image." 
"""
from __future__ import annotations

import argparse
import torch
import sys
import pathlib
import os
from PIL import Image


# Ensure repo root and internvl_chat subfolder are on sys.path so local package `internvl` can be imported
repo_root = pathlib.Path(__file__).resolve().parents[1]
repo_root_str = str(repo_root)
internvl_chat_path = str(repo_root.joinpath("internvl_chat"))
for p in (internvl_chat_path, repo_root_str):
    if p not in sys.path:
        sys.path.insert(0, p)
    # scripts/ is one level under repo root; add repo root to path
    repo_root = str(pathlib.Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def parse_args():
    p = argparse.ArgumentParser(description="Run inference from an InternVL checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint folder (e.g. work_dirs/.../checkpoint-170080)")
    p.add_argument("--image", default=None, help="Path to an image file to pass to the model")
    p.add_argument("--prompt", default="Hello, who are you?", help="Text prompt or chat input")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--device_map", default="auto", help="device_map for transformers (auto or 'cpu' or 'cuda:0')")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--no_low_cpu_mem", action="store_true", help="Disable low_cpu_mem_usage flag")
    return p.parse_args()


def build_transform(input_size: int = 448):
    try:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
    except Exception:
        raise RuntimeError("Please install torchvision to preprocess images: pip install torchvision")

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform


def dynamic_preprocess(image, image_size=448, min_num=1, max_num=12, use_thumbnail=False):
    # Minimal dynamic preprocess: split into tiles approximately
    orig_w, orig_h = image.size
    aspect = orig_w / orig_h

    # choose number of tiles between 1 and max_num (simple heuristic)
    num_tiles = 1
    for n in range(1, max_num + 1):
        if n * n <= max_num:
            num_tiles = n * n
    # Create one tile version by resizing and center-cropping grid
    resized = image.resize((int(image_size * (num_tiles ** 0.5)), int(image_size * (num_tiles ** 0.5)))) if num_tiles > 1 else image.resize((image_size, image_size))
    # For simplicity, when num_tiles == 1 return full image
    if num_tiles == 1:
        return [image]
    else:
        # fallback to splitting into equal square tiles
        tiles = []
        grid = int(num_tiles ** 0.5)
        tile_w = resized.width // grid
        tile_h = resized.height // grid
        for r in range(grid):
            for c in range(grid):
                box = (c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h)
                tiles.append(resized.crop(box).resize((image_size, image_size)))
        if use_thumbnail:
            tiles.append(image.resize((image_size, image_size)))
        return tiles


def load_image_as_pixel_values(image_path: str, input_size: int = 448):
    img = Image.open(image_path).convert("RGB")
    tiles = dynamic_preprocess(img, image_size=input_size, max_num=12, use_thumbnail=True)
    transform = build_transform(input_size=input_size)
    pixel_values = [transform(t) for t in tiles]
    pixel_values = torch.stack(pixel_values)  # [num_tiles, 3, H, W]
    return pixel_values


def main():
    args = parse_args()

    checkpoint = args.checkpoint
    if not os.path.isdir(checkpoint):
        print(f"Checkpoint folder not found: {checkpoint}")
        sys.exit(1)

    # show files for debugging
    print("Checkpoint contains:", os.listdir(checkpoint))

    # check safetensors
    model_files = [
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.pt",
    ]
    found = [f for f in model_files if os.path.exists(os.path.join(checkpoint, f))]
    if not found:
        print("No known model weight file found in checkpoint. Aborting.")
        sys.exit(1)
    if "model.safetensors" in found:
        try:
            import safetensors.torch  # noqa: F401
        except Exception:
            print("Checkpoint uses safetensors; install with: pip install safetensors")
            sys.exit(1)

    # dtype selection
    if args.dtype == "auto":
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # import local model classes to avoid HF dynamic import issues
    try:
        from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
    except Exception as e:
        print("Failed to import local InternVL classes:", e)
        print("Make sure you run this script from the repo root and PYTHONPATH includes the repo.")
        sys.exit(1)

    # load config & tokenizer from checkpoint
    config = InternVLChatConfig.from_pretrained(checkpoint)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)

    # load model
    low_cpu = not args.no_low_cpu_mem
    print(f"Loading model from {checkpoint} with dtype={dtype} device_map={args.device_map} low_cpu_mem_usage={low_cpu}")
    try:
        model = InternVLChatModel.from_pretrained(
            checkpoint,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=low_cpu,
            device_map=args.device_map,
        )
        model.eval()
    except Exception as e:
        print("Primary load failed:", e)
        print("Falling back to device_map=None (CPU load) and moving to GPU if available")
        model = InternVLChatModel.from_pretrained(
            checkpoint,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=low_cpu,
            device_map=None,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

    # prepare generation config
    generation_config = {"max_new_tokens": args.max_new_tokens, "do_sample": True}

    # Run chat / forward depending on presence of image and model API
    if args.image:
        # Build pixel_values compatible with model's expected image/token size.
        from PIL import Image

        # Load original image
        orig_img = Image.open(args.image).convert("RGB")

        # If the model has `config.force_image_size`, resize to that to match training preprocessing
        force_image_size = getattr(getattr(model, "config", None), "force_image_size", None)
        if force_image_size is None:
            # fallback to a sane default
            force_image_size = 448

        # Create a single tile resized image to avoid token-count mismatches
        resized = orig_img.resize((force_image_size, force_image_size))

        # Transform and stack into pixel_values [1, 3, H, W]
        transform = build_transform(input_size=force_image_size)
        pv = transform(resized)
        pixel_values = torch.stack([pv])

        # move to model device and cast
        device = next(model.parameters()).device
        pixel_values = pixel_values.to(device=device, dtype=next(model.parameters()).dtype)

        print(f"Running image+prompt chat with single tile resized to {force_image_size}x{force_image_size}...")
        try:
            out = model.chat(tokenizer, pixel_values, args.prompt, generation_config)
            print("Output:\n", out)
        except Exception as e:
            print("model.chat failed:", e)
            print("If this is a shape-mismatch, try verifying model.config.force_image_size, model.num_image_token, and that the model expects 1 image tile.")
    else:
        print("Running text-only chat/forward...")
        try:
            if hasattr(model, "chat"):
                out = model.chat(tokenizer, None, args.prompt, generation_config)
                print("Output:\n", out)
            else:
                input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
                with torch.no_grad():
                    res = model(input_ids)
                print("Forward pass OK")
        except Exception as e:
            print("Inference failed:", e)


if __name__ == "__main__":
    main()
