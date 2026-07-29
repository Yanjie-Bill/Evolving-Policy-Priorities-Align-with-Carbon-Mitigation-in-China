from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CLUSTER_COLORS, CLUSTER_LABELS, PROCESSED, RAW, ensure_output_dir, savefig, set_plot_style


def plot_radar(df: pd.DataFrame, outdir) -> None:
    cluster_cols = [str(i) for i in range(7)]
    type_means = df.groupby("ClusterType")[cluster_cols].mean().sort_index()
    angles = np.linspace(0, 2 * math.pi, len(cluster_cols), endpoint=False)
    angles = np.r_[angles, angles[0]]

    set_plot_style()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), subplot_kw={"projection": "polar"})
    axes = axes.ravel()
    for ax, (cluster_type, row) in zip(axes, type_means.iterrows()):
        values = row.to_numpy(dtype=float)
        values = np.r_[values, values[0]]
        ax.plot(angles, values, color="#333333", linewidth=1.5)
        ax.fill(angles, values, color="#4C78A8", alpha=0.25)
        ax.set_title(f"Type {int(cluster_type)}")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f"C{i}" for i in range(7)], fontsize=7)
        ax.set_ylim(0, max(1.0, type_means.max().max()))
        ax.set_yticklabels([])
    for ax in axes[len(type_means) :]:
        ax.axis("off")
    fig.suptitle("Provincial policy typology fingerprints", y=1.02)
    savefig(fig, outdir, "fig2a_typology_radar")


def plot_reduction_boxplot(df: pd.DataFrame, outdir) -> None:
    set_plot_style()
    df = df.dropna(subset=["ClusterType", "reduction_pct"]).copy()
    groups = [g["reduction_pct"].to_numpy() for _, g in df.groupby("ClusterType")]
    labels = [f"T{int(t)}" for t in sorted(df["ClusterType"].unique())]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.boxplot(groups, tick_labels=labels, patch_artist=True, boxprops={"facecolor": "#D8DEE9"})
    for i, (_, g) in enumerate(df.groupby("ClusterType"), start=1):
        jitter = np.random.default_rng(42 + i).normal(i, 0.035, len(g))
        ax.scatter(jitter, g["reduction_pct"], color="#333333", s=18, alpha=0.75)
    ax.axhline(df["reduction_pct"].mean(), color="#B73E3E", linestyle="--", linewidth=1)
    ax.set_ylabel("CO2/GDP reduction, 2019-2024 (%)")
    ax.set_xlabel("Policy typology")
    ax.set_title("Provincial carbon-intensity reduction by typology")
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, outdir, "fig2c_typology_reduction_boxplot")


def plot_typology_map(df: pd.DataFrame, outdir) -> None:
    try:
        import geopandas as gpd
    except Exception as exc:
        (outdir / "map_skipped.txt").write_text(
            f"Map skipped because geopandas is unavailable: {exc}\n", encoding="utf-8"
        )
        return

    shp = RAW / "ChinaAdminDivisonSHP-master" / "2. Province" / "province.shp"
    if not shp.exists():
        (outdir / "map_skipped.txt").write_text("Map skipped because province.shp is missing.\n", encoding="utf-8")
        return

    name_map = {
        "Anhui": "安徽省",
        "Beijing": "北京市",
        "Chongqing": "重庆市",
        "Fujian": "福建省",
        "Guangdong": "广东省",
        "Guizhou": "贵州省",
        "Hainan": "海南省",
        "Hebei": "河北省",
        "Heilongjiang": "黑龙江省",
        "Henan": "河南省",
        "Hunan": "湖南省",
        "Inner Mongolia": "内蒙古自治区",
        "Jiangsu": "江苏省",
        "Jiangxi": "江西省",
        "Jilin": "吉林省",
        "Liaoning": "辽宁省",
        "Ningxia": "宁夏回族自治区",
        "Qinghai": "青海省",
        "Shaanxi": "陕西省",
        "Shandong": "山东省",
        "Shanghai": "上海市",
        "Shanxi": "山西省",
        "Sichuan": "四川省",
        "Tianjin": "天津市",
        "Tibet": "西藏自治区",
        "Xinjiang": "新疆维吾尔自治区",
        "Yunnan": "云南省",
        "Zhejiang": "浙江省",
    }
    map_df = df.copy()
    map_df["pr_name"] = map_df["state_en"].map(name_map)
    gdf = gpd.read_file(shp)
    name_col = "pr_name" if "pr_name" in gdf.columns else gdf.columns[0]
    merged = gdf.merge(map_df, left_on=name_col, right_on="pr_name", how="left")

    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    merged.plot(column="ClusterType", categorical=True, legend=True, cmap="tab10", linewidth=0.4, edgecolor="white", ax=ax)
    ax.set_axis_off()
    ax.set_title("Provincial policy typology")
    savefig(fig, outdir, "fig2b_typology_map")


def main() -> None:
    outdir = ensure_output_dir("02_provincial_typology")
    typology = pd.read_csv(PROCESSED / "provincial" / "policy_emission_joint_typology.csv")
    reduction = pd.read_csv(PROCESSED / "provincial" / "typology_intensity_reduction_2019_2024.csv")
    plot_radar(typology, outdir)
    plot_reduction_boxplot(reduction, outdir)
    plot_typology_map(typology, outdir)
    typology.to_csv(outdir / "provincial_typology_table.csv", index=False)
    reduction.to_csv(outdir / "provincial_reduction_table.csv", index=False)
    print(f"Saved provincial typology outputs to {outdir}")


if __name__ == "__main__":
    main()
