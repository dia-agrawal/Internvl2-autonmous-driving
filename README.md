**InternVL3 for Autonomous Driving Scene Understanding**
**Second-Stage Finetuning + CLIP-Based Semantic Evaluation
This project adapts InternVL3-1B, a large vision-language model, to the autonomous-driving domain through driving-specific finetuning and introduces a CLIP-based semantic evaluation framework for image-text alignment scoring in traffic scenarios.**

**Overview**
Autonomous-driving datasets often contain noisy, weak, or inconsistent textual annotations. Standard VLM benchmarks fail to reflect visual correctness and GPT-judges overemphasize writing quality rather than factual grounding.
This project addresses those issues through:
Driving-domain finetuning of InternVL3-1B
Trained on DriveLM-style traffic reasoning JSONL data
Frozen InternViT vision encoder
Trainable projector + LLM
Improved reasoning about cars, pedestrians, traffic signs, intersections, hazards, etc.
A novel CLIP-based semantic evaluation framework
Computes cosine similarity between:
CLIP(text from InternVL output
CLIP(text from GT label) or CLIP(image)
Produces numeric, reproducible alignment scores
Handles weak labels and noisy datasets
Enables comparison across checkpoints and datasets
Together, these form a complete pipeline for training and evaluating multimodal driving-scene understanding models.

**Finetuning InternVL3-1B**
Training Setup

Base model: InternVL3-1B Chat
Frozen modules: InternViT image encoder
Trainable modules: projector + LLM
Input format: DriveLM-style JSONL
Loss: standard language-model cross entropy
Command
python finetune/train.py --config finetune/config.yaml (look at parameters offered such as amount of gpus or having to use grad acc) 

** CLIP-Based Semantic Evaluation (Novel Contribution)**
This evaluation method computes:
text–text similarity:
CLIP(InternVL output) vs CLIP(GT caption)
text–image similarity:
CLIP(InternVL output) vs CLIP(image features)
Command
python evaluation/clip_eval.py \
    --predictions results/generated.jsonl \
    --images data/test/images/ \
    --out results/scores.csv

Output metrics:
Cosine similarity (per-sample + dataset mean)
Per-dataset statistical summary

**Trained Model** https://drive.google.com/drive/folders/1VUlB1nOVD_DAi0iVhzj_SFVjNltgU92v?usp=sharing

Model comparison plots

This approach gives grounded, reproducible measurements even when labels are noisy, which is common in driving datasets.

** References & Base Code
**
This work builds upon the official InternVL repository:

InternVL:
https://github.com/OpenGVLab/InternVL
Other referenced works are cited in the accompanying 

**Setup** 
Use the requirement.txt to set up your enviornment. We are using Python 3.10

**To get the data and jsonl files, you have to look through the data load script and others**

