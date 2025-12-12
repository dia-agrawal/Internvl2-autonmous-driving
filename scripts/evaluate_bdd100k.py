#!/usr/bin/env python3
"""Evaluate InternVL3 on BDD100K dataset using val.jsonl.

Usage:
  python scripts/evaluate_bdd100k.py \
    --checkpoint work_dirs/.../checkpoint-170080 \
    --val_jsonl data/bdd100k/.../val.jsonl \
    --output_dir eval_results/

This script:
1. Loads the checkpoint (model + tokenizer).
2. Processes each record in the BDD100K JSONL (expects 'image' and optional 'question', 'answer' fields).
3. Generates model predictions.
4. Saves results to a JSON file and optionally computes metrics (BLEU, ROUGE, exact match if answers present).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import pathlib
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

# Add repo root and internvl_chat to path
repo_root = pathlib.Path(__file__).resolve().parents[1]
repo_root_str = str(repo_root)
internvl_chat_path = str(repo_root.joinpath("internvl_chat"))
for p in (internvl_chat_path, repo_root_str):
    if p not in sys.path:
        sys.path.insert(0, p)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate InternVL3 on BDD100K val.jsonl")
    p.add_argument("--checkpoint", type=str, default=None, help="Optional path to finetuned checkpoint (overrides default pretrained model)")
    p.add_argument("--pretrained_model_path", type=str, default="pretrained/InternVL3-1B", help="Path to pretrained model (default: pretrained/InternVL3-1B)")
    p.add_argument("--val_jsonl", required=True, help="Path to BDD100K val.jsonl")
    p.add_argument("--image_root", default=None, help="Root directory for images (e.g. data/bdd100k/bdd100k/bdd100k/images/100k/val). If None, inferred from val_jsonl dir.")
    p.add_argument("--output_dir", default="eval_results", help="Output directory for results")
    p.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate (for testing)")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    p.add_argument("--max_new_tokens", type=int, default=128, help="Max generation tokens")
    p.add_argument("--device_map", default="auto", help="Device map for model placement")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--compute_metrics", action="store_true", help="Compute BLEU/ROUGE if ground truth answers available")
    return p.parse_args()


def load_model_and_tokenizer(checkpoint, dtype="auto", device_map="auto"):
    """Load model and tokenizer from checkpoint."""
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
    
    print(f"Loading model from {checkpoint} with dtype={dtype_torch}")
    config = InternVLChatConfig.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)
    
    try:
        model = InternVLChatModel.from_pretrained(
            checkpoint,
            config=config,
            torch_dtype=dtype_torch,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
    except Exception as e:
        print(f"Primary load failed ({e}), falling back to device_map=None")
        model = InternVLChatModel.from_pretrained(
            checkpoint,
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
    """Build image transform pipeline."""
    try:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
    except Exception:
        raise RuntimeError("Please install torchvision: pip install torchvision")
    
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform


def load_image(image_path: str, force_image_size: int = 448):
    """Load and preprocess image to pixel_values."""
    from PIL import Image
    
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((force_image_size, force_image_size))
    transform = build_transform(input_size=force_image_size)
    pv = transform(img_resized)
    pixel_values = torch.stack([pv])  # [1, 3, H, W]
    return pixel_values


def generate_answer(model, tokenizer, pixel_values, prompt, max_new_tokens=128):
    """Generate model prediction for an image+prompt."""
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device=device, dtype=next(model.parameters()).dtype)
    
    generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
    
    with torch.no_grad():
        try:
            answer = model.chat(tokenizer, pixel_values, prompt, generation_config)
            return answer.strip()
        except Exception as e:
            return f"[Error: {str(e)[:100]}]"


def compute_metrics(predictions, references):
    """Compute simple metrics: exact match, BLEU (if available), ROUGE (if available)."""
    metrics = {}
    
    # Exact Match
    exact_matches = sum(1 for pred, ref in zip(predictions, references) if pred.lower() == ref.lower())
    metrics["exact_match"] = exact_matches / len(predictions) if predictions else 0.0
    
    # BLEU (if nltk available)
    try:
        from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
        
        # BLEU-4
        refs_tokenized = [[ref.lower().split()] for ref in references]
        preds_tokenized = [pred.lower().split() for pred in predictions]
        bleu_score = corpus_bleu(refs_tokenized, preds_tokenized)
        metrics["BLEU-4"] = bleu_score
    except ImportError:
        pass
    
    # ROUGE (if rouge available)
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        rouge1_scores = []
        rougeL_scores = []
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)
        
        metrics["ROUGE-1"] = np.mean(rouge1_scores)
        metrics["ROUGE-L"] = np.mean(rougeL_scores)
    except ImportError:
        pass
    
    return metrics


def main():
    args = parse_args()
    
    # Determine which model to use
    if args.checkpoint:
        model_path = args.checkpoint
        print(f"Using finetuned checkpoint: {model_path}")
    else:
        model_path = args.pretrained_model_path
        print(f"Using pretrained model: {model_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, dtype=args.dtype, device_map=args.device_map)
    force_image_size = getattr(getattr(model, "config", None), "force_image_size", 448)
    
    # Load JSONL
    print(f"Loading {args.val_jsonl}...")
    records = []
    with open(args.val_jsonl, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    if args.max_samples:
        records = records[:args.max_samples]
    
    print(f"Processing {len(records)} samples...")
    
    results = []
    predictions = []
    references = []
    
    # Determine image root directory
    if args.image_root:
        image_root = args.image_root
    else:
        # Default: assume images are in 'val/' subdirectory next to val.jsonl
        jsonl_dir = os.path.dirname(os.path.abspath(args.val_jsonl))
        image_root = os.path.join(jsonl_dir, "val")
        if not os.path.isdir(image_root):
            # Fallback: use same directory as JSONL
            image_root = jsonl_dir
    
    print(f"Using image root: {image_root}")
    
    for i, record in enumerate(tqdm(records, desc="Evaluating")):
        # Extract fields
        image_path = record.get("image")
        question = record.get("question", "Describe this image.")
        ground_truth = record.get("answer")

        # If no 'answer', try to extract from 'conversations' (DriveLM/Chat format)
        if ground_truth is None and "conversations" in record:
            for turn in record["conversations"]:
                role = turn.get("from", turn.get("role", "")).lower()
                if role in ("gpt", "assistant"):
                    ground_truth = turn.get("value", "").strip()
                    break
                # Optionally, update question from 'human' if not present
                if not record.get("question") and role in ("human", "user"):
                    question = turn.get("value", question)

        if not image_path:
            continue

        # Resolve image path (handle relative paths)
        if not os.path.isabs(image_path):
            # Try relative to image_root, then absolute, then other candidates
            candidates = [
                os.path.join(image_root, image_path),
                os.path.join(image_root, os.path.basename(image_path)),
                image_path,
            ]
            image_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

        if not os.path.exists(image_path):
            print(f"  Skipping {image_path} (not found)")
            continue

        # Load image and generate
        try:
            pixel_values = load_image(image_path, force_image_size=force_image_size)
            prompt = f"<image>\n{question}"
            prediction = generate_answer(model, tokenizer, pixel_values, prompt, args.max_new_tokens)
        except Exception as e:
            prediction = f"[Error: {str(e)[:100]}]"

        # Store result
        result = {
            "image": image_path,
            "question": question,
            "prediction": prediction,
        }
        if ground_truth:
            result["ground_truth"] = ground_truth
            references.append(ground_truth)

        results.append(result)
        predictions.append(prediction)
    
    # Save results
    results_path = os.path.join(args.output_dir, "bdd100k_eval_results_onlypt.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")
    
    # Compute metrics if ground truth available
    if args.compute_metrics and references:
        print("\nComputing metrics...")
        metrics = compute_metrics(predictions, references)
        metrics_path = os.path.join(args.output_dir, "bdd100k_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics: {metrics}")
        print(f"Saved metrics to {metrics_path}")
    
    # Summary
    print(f"\nEvaluation complete!")
    print(f"  Total records: {len(records)}")
    print(f"  Processed: {len(results)}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
