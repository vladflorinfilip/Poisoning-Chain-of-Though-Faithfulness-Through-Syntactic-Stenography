"""Plot the shared S1/voice SVD direction from saved JSON summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "sparse_autoencoders/artifacts/sae_svd_05b_l18"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def arrow(ax, angle: float, label: str, color: str, y_offset: float) -> None:
    end = np.array([math.cos(angle), math.sin(angle)])
    ax.annotate(
        "",
        xy=end,
        xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "lw": 2.4, "color": color},
    )
    ax.text(
        end[0] + 0.03,
        end[1] + y_offset,
        label,
        color=color,
        ha="left",
        va="center",
        fontweight="bold",
    )


def plot(results_dir: Path, output: Path, top_n: int) -> None:
    summary = json.loads((results_dir / "summary.json").read_text())
    features = json.loads((results_dir / "top_shared_sae_features.json").read_text())
    features = features[:top_n]
    style()

    fig = plt.figure(figsize=(13.2, 4.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.42])
    ax_geometry = fig.add_subplot(grid[0, 0])
    ax_singular = fig.add_subplot(grid[0, 1])
    ax_features = fig.add_subplot(grid[0, 2])

    # A: exact two-vector geometry implied by the task-mean cosine.
    cosine = float(summary["s1_voice_mean_cosine_sae_scaled"])
    angle = math.acos(np.clip(cosine, -1.0, 1.0))
    arrow(ax_geometry, angle / 2, "S1: stance 1 − 0", "#2878B5", 0.04)
    arrow(ax_geometry, -angle / 2, "Voice: active − passive", "#D95F02", -0.04)
    ax_geometry.annotate(
        "",
        xy=(1.02, 0),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "lw": 2.0, "color": "#333333"},
    )
    ax_geometry.text(
        0.53,
        0.035,
        "shared SVD direction",
        color="#333333",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax_geometry.text(
        0.02,
        -0.61,
        f"cosine = {cosine:.3f}\nangle = {math.degrees(angle):.1f}°",
        ha="left",
        va="bottom",
    )
    ax_geometry.set(
        xlim=(-0.05, 1.35),
        ylim=(-0.68, 0.68),
        aspect="equal",
        title="A  Task-mean alignment",
    )
    ax_geometry.axis("off")

    # B: task-average singular values in the two representations.
    labels = ["σ₁\nshared", "σ₂\ntask-specific"]
    residual = np.asarray(summary["residual_svd"]["task_singular_values"])
    sae = np.asarray(summary["sae_scaled_svd"]["task_singular_values"])
    x = np.arange(2)
    width = 0.34
    ax_singular.bar(x - width / 2, residual, width, color="#8A8A8A", label="Residual (896D)")
    ax_singular.bar(x + width / 2, sae, width, color="#2878B5", label="Scaled SAE (7,168D)")
    for offset, values in ((-width / 2, residual), (width / 2, sae)):
        for xpos, value in zip(x + offset, values):
            ax_singular.text(xpos, value + 0.035, f"{value:.2f}", ha="center", fontsize=8.5)
    ax_singular.set(
        xticks=x,
        xticklabels=labels,
        ylabel="Singular value",
        ylim=(0, 1.42),
        title="B  Shared versus task-specific signal",
    )
    ax_singular.grid(axis="y", linestyle=":", alpha=0.4)
    ax_singular.legend(frameon=False, fontsize=8.5, loc="upper right")

    # C: signed feature coefficients after orienting PC1 toward both bit-1 means.
    shown = list(reversed(features))
    loadings = np.asarray([row["loading"] for row in shown])
    feature_labels = [str(row["feature"]) for row in shown]
    colors = np.where(loadings >= 0, "#2878B5", "#D95F02")
    y = np.arange(len(shown))
    ax_features.barh(y, loadings, color=colors, height=0.72)
    ax_features.axvline(0, color="#555555", linewidth=0.8)
    ax_features.set(
        yticks=y,
        yticklabels=feature_labels,
        xlabel="Loading on shared SAE direction",
        ylabel="SAE feature ID",
        title=f"C  Top {top_n} shared-direction features",
    )
    ax_features.grid(axis="x", linestyle=":", alpha=0.4)
    ax_features.legend(
        handles=[
            Patch(facecolor="#2878B5", label="toward stance 1 / active"),
            Patch(facecolor="#D95F02", label="opposite loading"),
        ],
        frameon=False,
        fontsize=8.5,
        loc="upper left",
    )

    fig.suptitle(
        "A partially shared layer-18 direction links S1 stance and grammatical voice",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.025,
        "Qwen2.5-0.5B · frozen 7,168-unit SAE · 100 matched pairs per task · "
        "feature coordinates scaled by decoder-vector norm · SVD signs oriented toward bit 1",
        ha="center",
        fontsize=8.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()
    output = args.output or ROOT / "figures/shared_svd_summary"
    plot(args.results_dir, output, args.top_n)
    print(output.with_suffix(".png"))
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
