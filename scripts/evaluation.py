# scripts/evaluation.py

import os
import json
import argparse
from statistics import mean

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def load_samples(path):
    print(f"[INFO] Loading samples from: {path}")
    with open(path, "r") as f:
        content = f.read().strip()

    # Try Case 1: standard JSON (list or dict)
    try:
        if content.startswith("[") or content.startswith("{"):
            data = json.loads(content)
            if isinstance(data, dict):
                for key in ("results", "annotations", "data"):
                    if key in data and isinstance(data[key], list):
                        data = data[key]
                        break
            if isinstance(data, list):
                print(f"[INFO] Loaded {len(data)} samples (JSON).")
                return data
            else:
                raise ValueError("JSON must be a list of samples or a dict containing a list.")
    except json.JSONDecodeError:
        # Not a single JSON blob – treat as JSONL
        pass

    # Case 2: JSONL (one JSON per line)
    samples = []
    for i, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[WARN] Skipping bad JSONL line {i}: {e}")
    print(f"[INFO] Loaded {len(samples)} samples (JSONL).")
    return samples

    # Case 2: JSONL (one JSON per line)
    samples = []
    for i, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[WARN] Skipping bad JSONL line {i}: {e}")
    print(f"[INFO] Loaded {len(samples)} samples (JSONL).")
    return samples


def get_image_path(ex, img_root):
    # adjust these keys if your JSON is different
    img_key_candidates = ["image", "img", "img_path", "image_path"]
    rel_path = None
    for k in img_key_candidates:
        if k in ex:
            rel_path = ex[k]
            break
    if rel_path is None:
        raise KeyError(f"None of {img_key_candidates} found in example keys: {list(ex.keys())}")

    if img_root:
        return os.path.expanduser(os.path.join(img_root, rel_path))
    return os.path.expanduser(rel_path)


def get_caption(ex):
    cap_key_candidates = ["prediction", "caption", "output", "answer"]
    for k in cap_key_candidates:
        if k in ex:
            return ex[k]
    raise KeyError(f"None of {cap_key_candidates} found in example keys: {list(ex.keys())}")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading CLIP model: {args.clip_model}")
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    samples = load_samples(args.input)
    scores = []

    out_f = open(args.output, "w") if args.output else None
    max_n = args.max_samples if args.max_samples > 0 else len(samples)
    print(f"[INFO] Will evaluate up to {max_n} samples.")

    processed = 0
    for idx, ex in enumerate(samples[:max_n], start=1):
        try:
            img_path = get_image_path(ex, args.img_root)
            caption = get_caption(ex)
        except KeyError as e:
            print(f"[WARN] Sample {idx}: {e} — skipping.")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Sample {idx}: cannot open image {img_path}: {e} — skipping.")
            continue

        caption = caption[:512]

        inputs = processor(
            text=[caption],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,      # <-- important
            max_length=77         # CLIP text limit
        ).to(device)


        with torch.no_grad():
            outputs = model(**inputs)
            score = outputs.logits_per_image[0, 0].item()

        ex["clip_score_pred"] = score
        scores.append(score)
        processed += 1

        if out_f:
            out_f.write(json.dumps(ex) + "\n")

        if processed % 5 == 0:
            print(f"[INFO] Processed {processed} samples, avg CLIP score: {mean(scores):.3f}")

    if out_f:
        out_f.close()

    print(f"[INFO] Done. Successfully evaluated {processed} samples.")
    if scores:
        print(f"[INFO] Final avg CLIP score: {mean(scores):.3f}")
    else:
        print("[WARN] No scores computed (all samples skipped?).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON or JSONL file with model predictions")
    parser.add_argument("--output", default=None, help="Optional JSONL output file with CLIP scores")
    parser.add_argument("--img-root", default="", help="Root dir to prefix to image paths")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to eval (0 = all)")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    args = parser.parse_args()
    main(args)
