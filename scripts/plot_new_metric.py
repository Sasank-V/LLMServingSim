#!/usr/bin/env python3
"""
scripts/plot_new_metric.py

Visualizes the Tri-Concern Harmonized Utility Metric (TCHUM / FairRoute Index)
and its component metrics across policies and scenarios.

Outputs generated under the artifact directory:
  1. overall_leaderboard.png
  2. category_comparison.png
  3. component_breakdown.png
  4. scenario_heatmap.png
  5. pareto_tradeoff.png
"""

import argparse
import glob
import os
import re
import sys

# Auto-add local workspace .venv to sys.path if needed
_workspace_venv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv")
if os.path.exists(_workspace_venv):
    for site_pkg in glob.glob(os.path.join(_workspace_venv, "lib", "python*", "site-packages")):
        if site_pkg not in sys.path:
            sys.path.insert(0, site_pkg)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import importlib.util
_new_metric_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new-metric.py")
_spec = importlib.util.spec_from_file_location("new_metric", _new_metric_path)
new_metric = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(new_metric)

POLICY_CATEGORY = new_metric.POLICY_CATEGORY
POLICY_ORDER = new_metric.POLICY_ORDER
load_dataset = new_metric.load_dataset
compute_tri_concern_metric = new_metric.compute_tri_concern_metric
aggregate_leaderboard = new_metric.aggregate_leaderboard

# Styling palette
CATEGORY_COLORS = {
    "FairRoute (v2 / H5)": "#d32f2f",
    "FairRoute (Final)": "#e53935",
    "Combination-Form Hypothesis": "#ff8f00",
    "Naive Linear Triple": "#00897b",
    "Pairwise": "#0288d1",
    "Single-Signal": "#1976d2",
    "Prior Art": "#8e24aa",
    "Naive": "#757575",
}


def plot_overall_leaderboard(leaderboard: pd.DataFrame, output_path: str):
    """Generate horizontal bar chart of overall Tri-Concern Utility score for all policies."""
    plt.figure(figsize=(12, 10), dpi=300)
    df = leaderboard.sort_values(by="TriConcern_Utility", ascending=True)

    colors = [CATEGORY_COLORS.get(cat, "#333333") for cat in df["Category"]]
    bars = plt.barh(df["policy_name"], df["TriConcern_Utility"], color=colors, edgecolor="none", height=0.7)

    for bar, val in zip(bars, df["TriConcern_Utility"]):
        plt.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
                 va="center", ha="left", fontsize=9, fontweight="bold", color="#333333")

    plt.title("Tri-Concern Harmonized Utility Metric (TCHUM / FairRoute Index)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Utility Score (Geometric Mean Across Scenarios)", fontsize=12, labelpad=10)
    plt.ylabel("Routing Policy", fontsize=12)
    plt.xlim(0, max(df["TriConcern_Utility"]) * 1.15)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    # Custom legend for categories
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for cat, col in CATEGORY_COLORS.items() if cat in df["Category"].values]
    labels = [cat for cat in CATEGORY_COLORS.keys() if cat in df["Category"].values]
    plt.legend(handles, labels, title="Policy Category", loc="lower right", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_category_comparison(leaderboard: pd.DataFrame, output_path: str):
    """Generate bar chart comparing mean utility scores by policy category."""
    plt.figure(figsize=(10, 6), dpi=300)
    cat_df = leaderboard.groupby("Category")["TriConcern_Utility"].agg(["mean", "std", "count"]).reset_index()
    cat_df = cat_df.sort_values(by="mean", ascending=False)

    colors = [CATEGORY_COLORS.get(cat, "#333333") for cat in cat_df["Category"]]
    bars = plt.bar(cat_df["Category"], cat_df["mean"], color=colors, width=0.55, edgecolor="none")

    for bar, val in zip(bars, cat_df["mean"]):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.008, f"{val:.4f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.title("Mean Tri-Concern Utility Score by Policy Category", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Mean Tri-Concern Score", fontsize=12)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.ylim(0, max(cat_df["mean"]) * 1.18)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_component_breakdown(leaderboard: pd.DataFrame, output_path: str):
    """Generate grouped bar chart showing Locality, Perf/Load, and Fairness scores for top policies."""
    plt.figure(figsize=(14, 7), dpi=300)

    # Select representative subset of policies
    repr_policies = ["H3", "H2", "FAIRROUTE_V2", "FAIRROUTE", "FAIRNESS", "LOCALITY", "DUALMAP", "LBGR", "LOAD", "RR", "PREBLE", "NEXUSSCHED"]
    df = leaderboard[leaderboard["policy_name"].isin(repr_policies)].copy()
    df["policy_name"] = pd.Categorical(df["policy_name"], categories=repr_policies, ordered=True)
    df = df.sort_values("policy_name")

    x = np.arange(len(df))
    width = 0.25

    plt.bar(x - width, df["Mean_Locality_Score"], width, label="Locality Score (E_L)", color="#1e88e5")
    plt.bar(x, df["Mean_Perf_Score"], width, label="Perf / Load Score (E_P)", color="#43a047")
    plt.bar(x + width, df["Mean_Fairness_Score"], width, label="Fairness Score (E_F)", color="#e53935")

    plt.title("Component Score Breakdown Across Key Routing Policies", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Routing Policy", fontsize=12, labelpad=10)
    plt.ylabel("Component Score (0.0 - 1.0)", fontsize=12)
    plt.xticks(x, df["policy_name"], rotation=30, ha="right", fontsize=10)
    plt.ylim(0, 1.15)
    plt.legend(loc="upper right", frameon=True, facecolor="white")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_scenario_heatmap(summary: pd.DataFrame, output_path: str):
    """Generate heatmap of Tri-Concern Score per Scenario x Policy."""
    plt.figure(figsize=(15, 9), dpi=300)
    pivot = summary.pivot(index="scenario", columns="policy_name", values="Scenario_TriConcern_Score")

    # Order policies by overall rank
    ordered_cols = [p for p in POLICY_ORDER if p in pivot.columns]
    pivot = pivot.reindex(columns=ordered_cols)

    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={"label": "Scenario Tri-Concern Score"},
                annot_kws={"size": 7}, linewidths=0.3)

    plt.title("Scenario × Policy Heatmap of Tri-Concern Utility Scores", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Routing Policy", fontsize=12, labelpad=10)
    plt.ylabel("Workload Scenario", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_pareto_tradeoff(summary: pd.DataFrame, output_path: str):
    """Generate scatter plot showing Goodput vs Prefix Hit Ratio vs TTFT P99."""
    plt.figure(figsize=(11, 7), dpi=300)

    # Focus on full_adversarial scenario (highest conflict)
    adv = summary[summary["scenario"] == "full_adversarial"].copy()
    if adv.empty:
        adv = summary.groupby("policy_name").mean().reset_index()

    categories = adv["policy_name"].map(POLICY_CATEGORY).fillna("Other")

    for cat, color in CATEGORY_COLORS.items():
        sub = adv[categories == cat]
        if sub.empty:
            continue
        plt.scatter(sub["goodput_sps"], sub["mean_prefix_hit_ratio"],
                    s=np.clip(12000.0 / (sub["ttft_p99_ms"] + 10.0), 30, 400),
                    color=color, alpha=0.85, edgecolors="black", linewidth=0.7, label=cat)

        for _, row in sub.iterrows():
            plt.annotate(row["policy_name"], (row["goodput_sps"], row["mean_prefix_hit_ratio"]),
                         xytext=(4, 4), textcoords="offset points", fontsize=8, fontweight="bold")

    plt.title("Full Adversarial Trade-Off: Goodput vs. Locality (Bubble Size = 1 / TTFT Latency)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Goodput (req/s, higher better)", fontsize=11)
    plt.ylabel("Mean Prefix Cache Hit Ratio (higher better)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Policy Category", loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots for Tri-Concern Metric")
    parser.add_argument("--data-dir", type=str, default="outputs/fairroute_hypothesis",
                        help="Path to directory containing hypothesis CSV files")
    parser.add_argument("--output-dir", type=str,
                        default="/home/sriram/.gemini/antigravity/brain/8d747cb8-2fdd-410c-be7f-6e3dd8331ab0",
                        help="Directory to save generated plot images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.data_dir}...")
    requests_df = load_dataset(args.data_dir)
    summary = compute_tri_concern_metric(requests_df)
    leaderboard = aggregate_leaderboard(summary)

    print(f"Generating plot artifacts in {args.output_dir}...")
    plot_overall_leaderboard(leaderboard, os.path.join(args.output_dir, "overall_leaderboard.png"))
    plot_category_comparison(leaderboard, os.path.join(args.output_dir, "category_comparison.png"))
    plot_component_breakdown(leaderboard, os.path.join(args.output_dir, "component_breakdown.png"))
    plot_scenario_heatmap(summary, os.path.join(args.output_dir, "scenario_heatmap.png"))
    plot_pareto_tradeoff(summary, os.path.join(args.output_dir, "pareto_tradeoff.png"))

    print("All plots successfully generated!")


if __name__ == "__main__":
    main()
