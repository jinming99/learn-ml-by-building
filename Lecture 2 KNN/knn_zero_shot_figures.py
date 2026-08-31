"""Polished figures for the Lecture 2 distance-learning case study.

The notebook keeps the modeling code visible and delegates plotting details here
so students can focus on the ideas the figures are meant to explain.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


INK = "#172033"
MUTED = "#64748B"
GRID = "#D9E1EA"
TEAL = "#0F766E"
BLUE = "#2563EB"
GOLD = "#D97706"
PURPLE = "#7C3AED"


def set_case_study_style():
    """Apply one restrained visual system to all notebook figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "font.size": 11,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )


def plot_metric_rank_matrix(metric_ranks, query_label, k):
    """Show tie-aware top-k ranks under several scoring rules."""
    colors = ListedColormap([TEAL, "#67B7AE", "#C7E5E1"][:k])
    colors.set_bad("#F1F5F9")
    values = metric_ranks.to_numpy(dtype=float)
    image_values = np.ma.masked_invalid(values - 1)

    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    ax.imshow(image_values, cmap=colors, vmin=0, vmax=k - 1, aspect="auto")

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            rank = values[row, column]
            label = "—" if np.isnan(rank) else str(int(rank))
            color = "white" if rank == 1 else INK
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
            )

    ax.set_xticks(range(len(metric_ranks.columns)))
    ax.set_xticklabels(metric_ranks.columns)
    ax.set_yticks(range(len(metric_ranks.index)))
    ax.set_yticklabels(metric_ranks.index)
    ax.tick_params(length=0, pad=9)
    ax.set_title(
        f"Top-{k} neighborhoods change with the scoring rule",
        loc="left",
        pad=29,
    )
    ax.text(
        0,
        1.025,
        f"{query_label}; all eight features; cutoff ties retained",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
        va="bottom",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_metric_learning(
    gap_history,
    feedback_labels,
    features,
    feature_labels,
    learned_weights,
    margin,
    epochs_used,
):
    """Show training success and the resulting diagonal metric."""
    loss_history = np.maximum(0, margin - gap_history)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        gridspec_kw={"width_ratios": [1.08, 1]},
    )

    epoch_axis = np.arange(len(loss_history))
    for column, (label, color) in enumerate(
        zip(feedback_labels, [TEAL, BLUE, PURPLE])
    ):
        axes[0].plot(
            epoch_axis,
            loss_history[:, column],
            marker="o",
            markersize=4,
            linewidth=2,
            color=color,
            label=label,
        )

    axes[0].axhline(0, color=MUTED, linewidth=1)
    axes[0].set_xticks(epoch_axis)
    axes[0].set_xlabel("Training epoch")
    axes[0].set_ylabel(r"Hinge loss  $\max(0, m-g)$")
    axes[0].set_title("Constraint loss during training", loc="left")
    axes[0].legend(frameon=False, ncol=3, fontsize=9, loc="upper right")
    axes[0].grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)

    order = np.argsort(learned_weights)[::-1]
    ordered_weights = learned_weights[order]
    ordered_features = [features[index] for index in order]
    bar_colors = [TEAL if weight >= 1 else "#A8B3C2" for weight in ordered_weights]

    axes[1].barh(
        range(len(ordered_features)),
        ordered_weights,
        color=bar_colors,
        height=0.68,
    )
    axes[1].axvline(1, color=GOLD, linestyle="--", linewidth=1.5)
    axes[1].set_yticks(range(len(ordered_features)))
    axes[1].set_yticklabels(
        [feature_labels[name] for name in ordered_features],
        fontsize=9.5,
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Learned weight")
    axes[1].set_title("Feedback changes the geometry", loc="left")
    axes[1].grid(axis="x", color=GRID, linewidth=0.8, alpha=0.8)
    axes[1].text(
        1,
        -0.55,
        "equal-weight start",
        ha="center",
        va="bottom",
        fontsize=9,
        color=GOLD,
    )

    for row, weight in enumerate(ordered_weights):
        axes[1].text(
            weight + 0.025,
            row,
            f"{weight:.2f}",
            va="center",
            fontsize=9,
            color=INK,
        )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"All feedback constraints are satisfied after {epochs_used} epochs",
        fontsize=15,
        fontweight="bold",
        x=0.04,
        ha="left",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


def plot_compatibility(
    compatibility,
    students,
    query_index,
    alignment_features,
    coverage_features,
    target_gap,
):
    """Contrast zero-gap similarity with target-gap complementarity."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        gridspec_kw={"width_ratios": [1, 1.15]},
    )

    selected_features = alignment_features + coverage_features
    observed_gap_max = float(
        (students[selected_features].max() - students[selected_features].min()).max()
    )
    gap_axis = np.linspace(0, observed_gap_max, 300)
    axes[0].plot(
        gap_axis,
        gap_axis**2,
        color=TEAL,
        linewidth=2.5,
        label="alignment: target gap = 0",
    )
    axes[0].plot(
        gap_axis,
        (gap_axis - target_gap) ** 2,
        color=GOLD,
        linewidth=2.5,
        label=f"role coverage: target gap = {target_gap}",
    )
    axes[0].scatter([0, target_gap], [0, 0], s=55, color=[TEAL, GOLD], zorder=4)
    axes[0].axvline(target_gap, color=GOLD, linestyle=":", linewidth=1)
    axes[0].set_xlabel(f"Absolute feature gap (observed range: 0–{observed_gap_max:.1f})")
    axes[0].set_ylabel("Component penalty")
    axes[0].set_title("Different goals prefer different gaps", loc="left")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(color=GRID, linewidth=0.8, alpha=0.8)

    point_counts = (
        compatibility.groupby(["alignment_cost", "coverage_cost"])
        .size()
        .reset_index(name="candidate_count")
    )
    axes[1].scatter(
        point_counts["alignment_cost"],
        point_counts["coverage_cost"],
        s=42 + 24 * point_counts["candidate_count"],
        color="#B8C2CF",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
    )

    best_index = compatibility.index[0]
    target_gap_index = compatibility["coverage_cost"].idxmin()
    minimum_alignment = compatibility["alignment_cost"].min()
    redundant_index = (
        compatibility.loc[
            compatibility["alignment_cost"].eq(minimum_alignment)
        ]
        .sort_values(["coverage_cost", "student_id"], ascending=[False, True])
        .index[0]
    )

    highlights = [
        (best_index, "lowest designed cost", TEAL, (8, 8), "left"),
        (target_gap_index, "closest to coverage target", GOLD, (-8, 8), "right"),
        (redundant_index, "aligned, little role coverage", PURPLE, (8, 8), "left"),
    ]
    seen = set()
    for candidate_index, description, color, offset, alignment in highlights:
        if candidate_index in seen:
            continue
        seen.add(candidate_index)
        row = compatibility.loc[candidate_index]
        axes[1].scatter(
            row["alignment_cost"],
            row["coverage_cost"],
            s=95,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=4,
        )
        axes[1].annotate(
            f"{row['student_id']}: {description}",
            (row["alignment_cost"], row["coverage_cost"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            color=color,
            fontweight="bold",
            ha=alignment,
        )

    x_limit = max(0.4, point_counts["alignment_cost"].max() * 1.08)
    y_limit = max(0.4, point_counts["coverage_cost"].max() * 1.08)
    axes[1].text(
        0.98,
        0.98,
        "marker size = tied candidates",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=MUTED,
    )
    axes[1].text(
        0.04,
        0.06,
        "better balance",
        transform=axes[1].transAxes,
        fontsize=9,
        color=TEAL,
        fontweight="bold",
    )
    axes[1].set_xlim(-0.02, x_limit)
    axes[1].set_ylim(-0.02, y_limit)
    axes[1].set_xlabel("Alignment cost")
    axes[1].set_ylabel("Coverage cost")
    axes[1].set_title("Candidates trade off alignment and role coverage", loc="left")
    axes[1].grid(color=GRID, linewidth=0.8, alpha=0.55)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Complementarity targets a moderate gap",
        fontsize=15,
        fontweight="bold",
        x=0.04,
        ha="left",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()
