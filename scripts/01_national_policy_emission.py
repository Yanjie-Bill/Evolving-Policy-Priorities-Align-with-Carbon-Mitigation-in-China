from __future__ import annotations

import ast

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from common import CLUSTER_COLORS, CLUSTER_LABELS, PROCESSED, ensure_output_dir, savefig, set_plot_style


def load_fig1b_data() -> pd.DataFrame:
    path = PROCESSED / "national" / "delta_yoy_vs_cluster_freq_normalized_data.csv"
    df = pd.read_csv(path)
    for c in [str(i) for i in range(7)]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def plot_policy_emission_alignment(df: pd.DataFrame, outdir) -> None:
    set_plot_style()
    cluster_cols = [str(i) for i in range(7)]
    years = df["year"]
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.5), sharex=True, constrained_layout=True)
    slowdown_spans = [(2005, 2008), (2011, 2012), (2014, 2015), (2019, 2021), (2023, 2024)]

    for ax in axes:
        for start, end in slowdown_spans:
            ax.axvspan(start, end, color="#F2B8B5", alpha=0.25, linewidth=0)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    axes[0].plot(years, df["delta_yoy"], color="black", linewidth=2.0, marker="o", markersize=3.5)
    axes[0].axhline(0, color="0.55", linestyle="--", linewidth=0.8)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Delta YoY growth")
    axes[0].set_title("National emission deceleration and policy attention")

    for c in cluster_cols:
        ci = int(c)
        axes[1].plot(
            years,
            df[c],
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=CLUSTER_COLORS[ci],
            label=CLUSTER_LABELS[ci],
            alpha=0.85,
        )
    axes[1].set_ylabel("Cluster share")
    axes[1].legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))

    axes[2].stackplot(
        years,
        [df[c] for c in cluster_cols],
        labels=[CLUSTER_LABELS[int(c)] for c in cluster_cols],
        colors=[CLUSTER_COLORS[int(c)] for c in cluster_cols],
        alpha=0.82,
    )
    axes[2].set_ylabel("Stacked share")
    axes[2].set_xlabel("Year")
    axes[2].set_xlim(int(years.min()), int(years.max()))

    savefig(fig, outdir, "fig1b_national_policy_emission_alignment")


def plot_embedding_tsne(outdir) -> None:
    path = PROCESSED / "national" / "policies_with_cluster_labels.csv"
    df = pd.read_csv(path)
    sample = df.copy()
    embeddings = []
    for value in sample["embedding"]:
        embeddings.append(ast.literal_eval(value))
    X = PCA(n_components=50, random_state=42).fit_transform(embeddings)
    coords = TSNE(
        n_components=2,
        perplexity=35,
        init="pca",
        learning_rate="auto",
        max_iter=1500,
        random_state=42,
    ).fit_transform(X)
    sample["tsne_x"] = coords[:, 0]
    sample["tsne_y"] = coords[:, 1]
    sample[["time", "policy_name", "state", "carbon_similarity", "cluster", "tsne_x", "tsne_y"]].to_csv(
        outdir / "fig1a_tsne_coordinates.csv", index=False
    )

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    sc = ax.scatter(
        sample["tsne_x"],
        sample["tsne_y"],
        c=sample["carbon_similarity"],
        cmap="RdYlBu_r",
        s=11,
        alpha=0.80,
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Carbon semantic similarity")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Policy embedding landscape")
    savefig(fig, outdir, "fig1a_policy_embedding_tsne")


def write_alignment_summary(outdir) -> None:
    metrics_path = PROCESSED / "national" / "dominant_policy_alignment_metrics.csv"
    validation_path = PROCESSED / "national" / "cluster_validation_metrics.csv"
    stability_path = PROCESSED / "national" / "cluster_stability_ARI.csv"

    metrics = pd.read_csv(metrics_path)
    validation = pd.read_csv(validation_path)
    stability = pd.read_csv(stability_path)
    with open(outdir / "national_summary.txt", "w", encoding="utf-8") as f:
        f.write("Dominant policy alignment metrics\n")
        f.write(metrics.to_string(index=False))
        f.write("\n\nCluster validation metrics\n")
        f.write(validation.to_string(index=False))
        f.write("\n\nCluster stability metrics\n")
        f.write(stability.to_string(index=False))
        f.write("\n")


def main() -> None:
    outdir = ensure_output_dir("01_national_policy_emission")
    df = load_fig1b_data()
    plot_policy_emission_alignment(df, outdir)
    plot_embedding_tsne(outdir)
    write_alignment_summary(outdir)
    print(f"Saved national policy-emission outputs to {outdir}")


if __name__ == "__main__":
    main()
