#!/usr/bin/env python
import argparse
import json
import os
from statistics import mean, median, pstdev

import matplotlib.pyplot as plt


def load_samples_generic(path):
    """Load JSON or JSONL with per-line objects."""
    with open(path, "r") as f:
        content = f.read().strip()

    # Try single JSON (list or dict) first
    try:
        if content.startswith("{") or content.startswith("["):
            data = json.loads(content)
            if isinstance(data, dict):
                # if wrapped
                for key in ("results", "annotations", "data"):
                    if key in data and isinstance(data[key], list):
                        data = data[key]
                        break
            if isinstance(data, list):
                return data
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL
    samples = []
    for i, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[WARN] {path}: bad JSONL line {i}: {e}")
    return samples


def extract_clip_scores(samples, score_keys=("clip_score_pred", "clip_score")):
    scores = []
    for ex in samples:
        for k in score_keys:
            if k in ex:
                try:
                    scores.append(float(ex[k]))
                except (TypeError, ValueError):
                    pass
                break
    return scores


def compute_metrics(scores):
    if not scores:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(scores),
        "mean": mean(scores),
        "median": median(scores),
        "std": pstdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
    }


def plot_mean_bar(summary, out_dir):
    files = []
    means = []
    for path, m in summary.items():
        if m["count"] > 0 and m["mean"] is not None:
            files.append(os.path.basename(path))
            means.append(m["mean"])

    if not files:
        print("[WARN] No data for mean bar plot.")
        return

    plt.figure()
    plt.bar(range(len(files)), means)
    plt.xticks(range(len(files)), files, rotation=30, ha="right")
    plt.ylabel("Mean CLIP score")
    plt.title("Mean CLIP score per file")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "clip_means_bar.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved mean bar plot to {out_path}")


def plot_boxplot(all_scores, out_dir):
    labels = []
    data = []
    for path, scores in all_scores.items():
        if scores:
            labels.append(os.path.basename(path))
            data.append(scores)

    if not data:
        print("[WARN] No data for boxplot.")
        return

    plt.figure()
    plt.boxplot(data, labels=labels, vert=True, showfliers=False)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("CLIP score")
    plt.title("CLIP score distribution per file")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "clip_boxplot.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved boxplot to {out_path}")


def plot_histograms(all_scores, out_dir):
    for path, scores in all_scores.items():
        if not scores:
            continue
        base = os.path.basename(path)
        safe_base = os.path.splitext(base)[0]

        plt.figure()
        plt.hist(scores, bins=30)
        plt.xlabel("CLIP score")
        plt.ylabel("Count")
        plt.title(f"CLIP score histogram: {base}")
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{safe_base}_hist.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved histogram for {base} to {out_path}")


def main(args):
    summary = {}
    all_scores = {}

    # Determine default plot dir
    if args.plot_dir:
        plot_dir = args.plot_dir
    else:
        # use dir of first file or current
        first = args.inputs[0]
        plot_dir = os.path.dirname(first) or "."
    os.makedirs(plot_dir, exist_ok=True)

    for path in args.inputs:
        if not os.path.isfile(path):
            print(f"[WARN] Skipping missing file: {path}")
            continue

        print(f"[INFO] Loading {path}")
        samples = load_samples_generic(path)
        scores = extract_clip_scores(samples)
        all_scores[path] = scores
        metrics = compute_metrics(scores)
        summary[path] = metrics

        if metrics["count"] > 0:
            print(
                f"[INFO] {path}: {metrics['count']} scores, "
                f"mean={metrics['mean']:.3f} "
                f"(min={metrics['min']:.3f}, max={metrics['max']:.3f})"
            )
        else:
            print(f"[WARN] {path}: no valid clip scores found")

    # Comparison table, sorted by mean
    print("\n=== CLIP Score Comparison (by mean, desc) ===")
    sortable = [
        (path, m) for path, m in summary.items()
        if m["count"] > 0 and m["mean"] is not None
    ]
    sortable.sort(key=lambda x: x[1]["mean"], reverse=True)

    header = f"{'File':40} {'N':>6} {'Mean':>8} {'Med':>8} {'Std':>8} {'Min':>8} {'Max':>8}"
    print(header)
    print("-" * len(header))
    for path, m in sortable:
        print(
            f"{os.path.basename(path):40} "
            f"{m['count']:6d} "
            f"{m['mean']:8.3f} "
            f"{m['median']:8.3f} "
            f"{m['std']:8.3f} "
            f"{m['min']:8.3f} "
            f"{m['max']:8.3f}"
        )

    # JSON summary
    if args.output:
        out = {
            "files": summary,
            "ranking_by_mean": [p for p, _ in sortable],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[INFO] Wrote summary to {args.output}")

    # Plots
    plot_mean_bar(summary, plot_dir)
    plot_boxplot(all_scores, plot_dir)
    plot_histograms(all_scores, plot_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of JSON/JSONL metrics files to compare",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file to save per-file metrics + ranking",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Directory to save plots (default: dir of first input)",
    )
    args = parser.parse_args()
    main(args)
