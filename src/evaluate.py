"""Evaluation script to aggregate metrics and generate comparison plots."""

import os
import sys
import json
import argparse
import wandb
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any


def parse_args():
    """Parse command line arguments."""
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: evaluate.py called with Hydra-style arguments (key=value) but uses argparse expecting --key value
    # [CAUSE]: Workflow calls `uv run python -u -m src.evaluate results_dir="..." run_ids='...'` without -- prefixes
    # [FIX]: Pre-process sys.argv to convert Hydra-style arguments to argparse format
    #
    # [OLD CODE]:
    # parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    # parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    # ...
    # return parser.parse_args()
    #
    # [NEW CODE]:
    # Convert key=value arguments to --key value format
    processed_argv = []
    for arg in sys.argv[1:]:
        if "=" in arg and not arg.startswith("--"):
            # Split key=value into --key value
            key, value = arg.split("=", 1)
            processed_argv.extend([f"--{key}", value])
        else:
            processed_argv.append(arg)

    parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Results directory"
    )
    parser.add_argument(
        "--run_ids", type=str, required=True, help="JSON list of run IDs to compare"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="airas", help="WandB entity"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="ui-test-20260303-v1", help="WandB project"
    )
    return parser.parse_args(processed_argv)


def fetch_run_data(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch run data from WandB API.

    Returns:
        Dictionary with 'config', 'summary', and 'history' keys
    """
    api = wandb.Api()

    # Find the most recent run with this display name
    runs = api.runs(
        f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    )

    if len(runs) == 0:
        print(f"WARNING: No run found with display_name={run_id}")
        return None

    run = runs[0]

    # Get config and summary
    config = run.config
    summary = dict(run.summary)

    # Get history (time series metrics)
    history = run.history()
    history_dict = history.to_dict("records") if len(history) > 0 else []

    return {
        "config": config,
        "summary": summary,
        "history": history_dict,
        "url": run.url,
    }


def export_per_run_metrics(results_dir: str, run_id: str, data: Dict[str, Any]) -> None:
    """Export per-run metrics to JSON."""
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    metrics = {
        "run_id": run_id,
        "summary": data["summary"],
        "config": data["config"],
        "url": data["url"],
    }

    metrics_file = os.path.join(run_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Exported metrics: {metrics_file}")


def generate_per_run_figures(
    results_dir: str, run_id: str, data: Dict[str, Any]
) -> None:
    """Generate per-run figures."""
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Since this is inference-only (not iterative training), we have limited time-series data
    # Generate a simple summary bar chart
    summary = data["summary"]

    # Extract key metrics
    metrics_to_plot = {}
    for key in ["accuracy", "avg_tokens", "cot_trigger_rate", "confident_wrong_rate"]:
        if key in summary:
            metrics_to_plot[key] = summary[key]

    if len(metrics_to_plot) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))

        metric_names = list(metrics_to_plot.keys())
        metric_values = list(metrics_to_plot.values())

        ax.barh(metric_names, metric_values, color="steelblue")
        ax.set_xlabel("Value")
        ax.set_title(f"Summary Metrics: {run_id}")
        ax.set_xlim(0, max(metric_values) * 1.1)

        plt.tight_layout()
        fig_file = os.path.join(run_dir, "summary_metrics.pdf")
        plt.savefig(fig_file)
        plt.close()

        print(f"Generated figure: {fig_file}")


def generate_comparison_plots(
    results_dir: str, run_ids: List[str], all_data: Dict[str, Dict]
) -> None:
    """Generate comparison plots across all runs."""
    comparison_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Extract metrics for all runs
    metrics_by_run = {}
    for run_id in run_ids:
        if run_id in all_data and all_data[run_id] is not None:
            metrics_by_run[run_id] = all_data[run_id]["summary"]

    if len(metrics_by_run) == 0:
        print("WARNING: No valid runs found for comparison")
        return

    # Identify common metrics
    all_metric_keys = set()
    for metrics in metrics_by_run.values():
        all_metric_keys.update(metrics.keys())

    # Filter to numeric metrics that exist in all runs
    common_metrics = []
    for key in all_metric_keys:
        values = [metrics_by_run[rid].get(key) for rid in metrics_by_run]
        if all(v is not None and isinstance(v, (int, float)) for v in values):
            common_metrics.append(key)

    print(f"Common metrics for comparison: {common_metrics}")

    # Plot 1: Bar chart comparing key metrics
    key_metrics = [
        m
        for m in ["accuracy", "avg_tokens", "cot_trigger_rate", "confident_wrong_rate"]
        if m in common_metrics
    ]

    if len(key_metrics) > 0:
        fig, axes = plt.subplots(1, len(key_metrics), figsize=(4 * len(key_metrics), 5))
        if len(key_metrics) == 1:
            axes = [axes]

        for idx, metric in enumerate(key_metrics):
            ax = axes[idx]

            run_ids_sorted = sorted(metrics_by_run.keys())
            values = [metrics_by_run[rid][metric] for rid in run_ids_sorted]

            # Color proposed method differently
            colors = [
                "coral" if "proposed" in rid else "steelblue" for rid in run_ids_sorted
            ]

            ax.bar(range(len(run_ids_sorted)), values, color=colors)
            ax.set_xticks(range(len(run_ids_sorted)))
            ax.set_xticklabels(
                [rid.replace("-", "\n") for rid in run_ids_sorted], fontsize=8
            )
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(metric.replace("_", " ").title())
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig_file = os.path.join(comparison_dir, "comparison_metrics.pdf")
        plt.savefig(fig_file)
        plt.close()

        print(f"Generated comparison: {fig_file}")

    # Plot 2: Scatter plot (accuracy vs tokens)
    if "accuracy" in common_metrics and "avg_tokens" in common_metrics:
        fig, ax = plt.subplots(figsize=(8, 6))

        for run_id in metrics_by_run:
            acc = metrics_by_run[run_id]["accuracy"]
            tokens = metrics_by_run[run_id]["avg_tokens"]

            color = "coral" if "proposed" in run_id else "steelblue"
            marker = "o" if "proposed" in run_id else "^"

            ax.scatter(
                tokens, acc, s=150, color=color, marker=marker, alpha=0.7, label=run_id
            )

        ax.set_xlabel("Average Tokens", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Accuracy vs Token Cost", fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_file = os.path.join(comparison_dir, "accuracy_vs_tokens.pdf")
        plt.savefig(fig_file, bbox_inches="tight")
        plt.close()

        print(f"Generated comparison: {fig_file}")


def export_aggregated_metrics(
    results_dir: str, run_ids: List[str], all_data: Dict[str, Dict]
) -> None:
    """Export aggregated comparison metrics."""
    comparison_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Collect metrics by run
    metrics_by_run = {}
    for run_id in run_ids:
        if run_id in all_data and all_data[run_id] is not None:
            metrics_by_run[run_id] = all_data[run_id]["summary"]

    # Identify proposed vs baseline runs
    proposed_runs = [rid for rid in metrics_by_run if "proposed" in rid]
    baseline_runs = [rid for rid in metrics_by_run if "proposed" not in rid]

    # Compute best in each category
    primary_metric = "accuracy"

    best_proposed = None
    best_proposed_value = -float("inf")
    if proposed_runs:
        for rid in proposed_runs:
            if primary_metric in metrics_by_run[rid]:
                val = metrics_by_run[rid][primary_metric]
                if val > best_proposed_value:
                    best_proposed_value = val
                    best_proposed = rid

    best_baseline = None
    best_baseline_value = -float("inf")
    if baseline_runs:
        for rid in baseline_runs:
            if primary_metric in metrics_by_run[rid]:
                val = metrics_by_run[rid][primary_metric]
                if val > best_baseline_value:
                    best_baseline_value = val
                    best_baseline = rid

    gap = None
    if best_proposed is not None and best_baseline is not None:
        gap = best_proposed_value - best_baseline_value

    aggregated = {
        "primary_metric": primary_metric,
        "metrics": metrics_by_run,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
    }

    agg_file = os.path.join(comparison_dir, "aggregated_metrics.json")
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Exported aggregated metrics: {agg_file}")

    if gap is not None:
        print(f"\nComparison Summary:")
        print(
            f"  Best Proposed ({best_proposed}): {primary_metric}={best_proposed_value:.3f}"
        )
        print(
            f"  Best Baseline ({best_baseline}): {primary_metric}={best_baseline_value:.3f}"
        )
        print(f"  Gap: {gap:+.3f}")


def main():
    """Main evaluation function."""
    args = parse_args()

    # Parse run_ids from JSON string
    run_ids = json.loads(args.run_ids)
    print(f"Evaluating runs: {run_ids}")

    # Fetch data from WandB for each run
    all_data = {}
    for run_id in run_ids:
        print(f"\nFetching data for {run_id}...")
        data = fetch_run_data(args.wandb_entity, args.wandb_project, run_id)
        all_data[run_id] = data

        if data is not None:
            # Export per-run metrics
            export_per_run_metrics(args.results_dir, run_id, data)

            # Generate per-run figures
            generate_per_run_figures(args.results_dir, run_id, data)

    # Generate comparison plots
    print(f"\nGenerating comparison plots...")
    generate_comparison_plots(args.results_dir, run_ids, all_data)

    # Export aggregated metrics
    export_aggregated_metrics(args.results_dir, run_ids, all_data)

    print(f"\nEvaluation complete. Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
