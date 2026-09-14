#!/usr/bin/env python3
"""
Plot Client Scalability Experiments for LLMServingSim

Analyzes how routing policies scale as the number of concurrent clients increases
(e.g., 5, 10, 20, 50, 100, 200 clients).

Inputs:
    CSV result files from scripts/run_client_sweep.sh matching:
    clients_<num_clients>_<scenario>_policy_<policy>.csv

Outputs:
    Publication-quality plots (PNG + PDF) saved to --output-dir:
    - client_count_vs_jain_fairness.png / .pdf
    - client_count_vs_p99_ttft.png / .pdf
    - client_count_vs_prefix_hit_ratio.png / .pdf
    - client_count_vs_p99_latency.png / .pdf
    - client_count_vs_slo_attainment.png / .pdf
    - client_scalability_summary_matrix.png / .pdf (2x3 multi-panel paper figure)
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Aesthetics & Color Palette
# ---------------------------------------------------------------------------

POLICY_COLORS = {
    "FAIRROUTE_V2": "#d32f2f",  # Bold Crimson / Red (Highlight)
    "H5": "#e53935",
    "FAIRROUTE": "#f44336",
    "H4": "#ef5350",
    "LOAD": "#1976d2",          # Blue (vLLM Baseline)
    "LOCALITY": "#388e3c",      # Green
    "FAIRNESS": "#7b1fa2",      # Purple
    "PREBLE": "#f57c00",        # Orange
    "H0": "#0288d1",
    "H1": "#0097a7",
    "RR": "#757575",            # Grey
}

POLICY_MARKERS = {
    "FAIRROUTE_V2": "o",
    "H5": "o",
    "FAIRROUTE": "s",
    "H4": "s",
    "LOAD": "^",
    "LOCALITY": "D",
    "FAIRNESS": "v",
    "PREBLE": "p",
    "H0": "X",
    "H1": "P",
    "RR": "*",
}

POLICY_LINESTYLES = {
    "FAIRROUTE_V2": "-",
    "H5": "-",
    "FAIRROUTE": "--",
    "H4": "--",
    "LOAD": "-",
    "LOCALITY": "-",
    "FAIRNESS": "-",
    "PREBLE": "-.",
    "H0": ":",
    "H1": ":",
    "RR": ":",
}


def load_client_sweep_data(input_dir: str) -> pd.DataFrame:
    """Load all CSV result files matching clients_<num>_<scenario>_policy_<policy>.csv."""
    pattern = os.path.join(input_dir, "clients_*_policy_*.csv")
    files = glob.glob(pattern)
    files = [f for f in files if not f.endswith("_timeseries.csv")]

    if not files:
        # Check subdirectories
        pattern_sub = os.path.join(input_dir, "**", "clients_*_policy_*.csv")
        files = glob.glob(pattern_sub, recursive=True)
        files = [f for f in files if not f.endswith("_timeseries.csv")]

    if not files:
        raise FileNotFoundError(f"No client sweep CSV files found in {input_dir}")

    filename_regex = re.compile(
        r"^clients_(?P<num_clients>\d+)_(?P<scenario>.+)_policy_(?P<policy>[A-Z0-9_]+)\.csv$"
    )

    records = []
    for filepath in files:
        fname = os.path.basename(filepath)
        match = filename_regex.match(fname)
        if not match:
            continue

        num_clients = int(match.group("num_clients"))
        scenario = match.group("scenario")
        policy = match.group("policy")

        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()

        # Latencies (convert ns -> ms)
        ttft = (df["TTFT"] if "TTFT" in df else df["ttft"]) / 1e6
        tpot = (df["TPOT"] if "TPOT" in df else df["tpot"]) / 1e6
        lat = (df["latency"] if "latency" in df else (df["end_time"] - df["arrival"])) / 1e6

        # Jain's Index over per-client latency
        client_col = "client id" if "client id" in df else ("client_id" if "client_id" in df else None)
        if client_col and client_col in df:
            client_lat = df.groupby(client_col)["latency"].mean()
            if len(client_lat) > 0 and (client_lat**2).sum() > 0:
                jain_lat = (client_lat.sum() ** 2) / (len(client_lat) * (client_lat**2).sum())
            else:
                jain_lat = 1.0
        else:
            jain_lat = 1.0

        # Cache Hit Ratio
        hit_col = "prefix_cache_hit" if "prefix_cache_hit" in df else "hit"
        hit_ratio = df[hit_col].mean() if hit_col in df else 0.0

        # SLO Attainment (% requests with TTFT < 50ms)
        slo_attainment = (ttft < 50.0).mean() * 100.0

        # Replica Load CV
        if "instance id" in df:
            inst_counts = df["instance id"].value_counts()
            load_cv = (inst_counts.std() / inst_counts.mean()) if inst_counts.mean() > 0 else 0.0
        else:
            load_cv = 0.0

        records.append({
            "num_clients": num_clients,
            "scenario": scenario,
            "policy": policy,
            "n_reqs": len(df),
            "jain_lat": jain_lat,
            "hit_ratio": hit_ratio,
            "mean_ttft_ms": ttft.mean(),
            "p99_ttft_ms": ttft.quantile(0.99),
            "mean_tpot_ms": tpot.mean(),
            "mean_lat_ms": lat.mean(),
            "p99_lat_ms": lat.quantile(0.99),
            "slo_attainment_pct": slo_attainment,
            "load_cv": load_cv,
        })

    summary_df = pd.DataFrame(records)
    print(f"Loaded {len(summary_df)} evaluations across client counts: {sorted(summary_df['num_clients'].unique())}")
    return summary_df


# ---------------------------------------------------------------------------
# Plotting Utilities
# ---------------------------------------------------------------------------


def plot_single_metric(
    df: pd.DataFrame,
    metric_col: str,
    y_label: str,
    title: str,
    output_path_base: str,
    higher_is_better: bool = True,
    log_scale: bool = False,
):
    """Plot a single metric vs. Number of Clients across policies."""
    plt.figure(figsize=(8, 5.5), dpi=300)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Average metric across scenarios per (num_clients, policy)
    avg_df = df.groupby(["num_clients", "policy"], as_index=False)[metric_col].mean()
    policies = sorted(avg_df["policy"].unique(), key=lambda p: (0 if p in ("FAIRROUTE_V2", "H5") else 1, p))

    for pol in policies:
        sub = avg_df[avg_df["policy"] == pol].sort_values("num_clients")
        color = POLICY_COLORS.get(pol, "#616161")
        marker = POLICY_MARKERS.get(pol, "o")
        linestyle = POLICY_LINESTYLES.get(pol, "-")
        linewidth = 2.5 if pol in ("FAIRROUTE_V2", "H5") else 1.8

        plt.plot(
            sub["num_clients"],
            sub[metric_col],
            label=pol,
            color=color,
            marker=marker,
            markersize=7,
            linestyle=linestyle,
            linewidth=linewidth,
        )

    plt.xlabel("Number of Concurrent Clients", fontsize=12, fontweight="bold")
    plt.ylabel(y_label, fontsize=12, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)

    if log_scale:
        plt.yscale("log")

    client_ticks = sorted(avg_df["num_clients"].unique())
    plt.xticks(client_ticks, [str(c) for c in client_ticks])

    plt.legend(title="Routing Policy", frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    plt.tight_layout()

    out_png = f"{output_path_base}.png"
    out_pdf = f"{output_path_base}.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


def plot_summary_matrix(df: pd.DataFrame, output_path_base: str):
    """Generate a 2x3 multi-panel paper summary grid."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=300)
    sns.set_theme(style="whitegrid")

    avg_df = df.groupby(["num_clients", "policy"], as_index=False).mean(numeric_only=True)
    policies = sorted(avg_df["policy"].unique(), key=lambda p: (0 if p in ("FAIRROUTE_V2", "H5") else 1, p))
    client_ticks = sorted(avg_df["num_clients"].unique())

    metrics_config = [
        ("jain_lat", "Jain's Fairness Index", "Fairness (Jain's Index)", False),
        ("p99_ttft_ms", "P99 TTFT (ms)", "Tail Latency (P99 TTFT)", True),
        ("hit_ratio", "Prefix Cache Hit Ratio (%)", "Prefix Cache Reuse", False),
        ("p99_lat_ms", "P99 End-to-End Latency (ms)", "Tail End-to-End Latency", True),
        ("slo_attainment_pct", "SLO Attainment % (TTFT < 50ms)", "SLO Attainment Rate", False),
        ("load_cv", "Replica Load Imbalance (CV)", "Replica Load Imbalance", False),
    ]

    for ax, (metric, y_label, title, is_log) in zip(axes.flat, metrics_config):
        for pol in policies:
            sub = avg_df[avg_df["policy"] == pol].sort_values("num_clients")
            color = POLICY_COLORS.get(pol, "#616161")
            marker = POLICY_MARKERS.get(pol, "o")
            linestyle = POLICY_LINESTYLES.get(pol, "-")
            linewidth = 2.4 if pol in ("FAIRROUTE_V2", "H5") else 1.6

            ax.plot(
                sub["num_clients"],
                sub[metric],
                label=pol,
                color=color,
                marker=marker,
                markersize=5.5,
                linestyle=linestyle,
                linewidth=linewidth,
            )

        ax.set_xlabel("Number of Concurrent Clients", fontsize=10, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(client_ticks)
        ax.set_xticklabels([str(c) for c in client_ticks], fontsize=9)

        if is_log:
            ax.set_yscale("log")

    # Add shared legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(policies),
        frameon=True,
        facecolor="white",
        edgecolor="0.8",
        fontsize=10,
    )

    plt.suptitle("Client Scalability Benchmark: Scaling Concurrent Clients (N=5 to 200)", y=1.06, fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_png = f"{output_path_base}.png"
    out_pdf = f"{output_path_base}.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Multi-Panel Summary: {out_png}")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="outputs/client_sweep",
        help="Directory containing client sweep CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/client_sweep/plots",
        help="Directory to save generated plots.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading client sweep results from: {args.input_dir} ...")
    df = load_client_sweep_data(args.input_dir)

    # 1. Client Count vs. Jain's Fairness Index
    plot_single_metric(
        df,
        metric_col="jain_lat",
        y_label="Jain's Fairness Index (Latency)",
        title="Client Scalability: Client Latency Fairness vs. Number of Clients",
        output_path_base=str(out_dir / "client_count_vs_jain_fairness"),
        higher_is_better=True,
    )

    # 2. Client Count vs. P99 TTFT
    plot_single_metric(
        df,
        metric_col="p99_ttft_ms",
        y_label="P99 Time to First Token (ms)",
        title="Client Scalability: Tail TTFT Latency vs. Number of Clients",
        output_path_base=str(out_dir / "client_count_vs_p99_ttft"),
        higher_is_better=False,
        log_scale=False,
    )

    # 3. Client Count vs. Prefix Cache Hit Ratio
    plot_single_metric(
        df,
        metric_col="hit_ratio",
        y_label="Prefix Cache Hit Ratio (%)",
        title="Client Scalability: Prefix Cache Reuse vs. Number of Clients",
        output_path_base=str(out_dir / "client_count_vs_prefix_hit_ratio"),
        higher_is_better=True,
    )

    # 4. Client Count vs. P99 End-to-End Latency
    plot_single_metric(
        df,
        metric_col="p99_lat_ms",
        y_label="P99 End-to-End Latency (ms)",
        title="Client Scalability: P99 Request Latency vs. Number of Clients",
        output_path_base=str(out_dir / "client_count_vs_p99_latency"),
        higher_is_better=False,
    )

    # 5. Client Count vs. SLO Attainment Rate
    plot_single_metric(
        df,
        metric_col="slo_attainment_pct",
        y_label="SLO Attainment Rate (% TTFT < 50ms)",
        title="Client Scalability: SLO Compliance vs. Number of Clients",
        output_path_base=str(out_dir / "client_count_vs_slo_attainment"),
        higher_is_better=True,
    )

    # 6. Multi-panel Summary Matrix for Paper
    plot_summary_matrix(df, str(out_dir / "client_scalability_summary_matrix"))

    print(f"\nAll plots generated successfully in: {out_dir}")


if __name__ == "__main__":
    main()
