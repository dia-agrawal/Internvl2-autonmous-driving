✅ FINAL README for Your GitHub Repo (Paste-Ready)
InternVL3 for Autonomous Driving Scene Understanding

Second-Stage Finetuning + CLIP-Based Semantic Evaluation

This project adapts InternVL3-1B, a large vision-language model, to the autonomous-driving domain through driving-specific finetuning and introduces a CLIP-based semantic evaluation framework for image-text alignment scoring in traffic scenarios.

🚗 Overview

Autonomous-driving datasets often contain noisy, weak, or inconsistent textual annotations. Standard VLM benchmarks fail to reflect visual correctness and GPT-judges overemphasize writing quality rather than factual grounding.

This project addresses those issues through:

Driving-domain finetuning of InternVL3-1B

Trained on DriveLM-style traffic reasoning JSONL data

Frozen InternViT vision encoder

Trainable projector + LLM

Improved reasoning about cars, pedestrians, traffic signs, intersections, hazards, etc.

A novel CLIP-based semantic evaluation framework

Computes cosine similarity between:

CLIP(text from InternVL output)

CLIP(text from GT label) or CLIP(image)

Produces numeric, reproducible alignment scores

Handles weak labels and noisy datasets

Enables comparison across checkpoints and datasets

Together, these form a complete pipeline for training and evaluating multimodal driving-scene understanding models.
