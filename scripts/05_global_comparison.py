from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from common import PROCESSED, ensure_output_dir, savefig, set_plot_style


COUNTRIES = ["China", "US", "India", "Russia", "Japan", "World"]
COUNTRY_NAME_MAP = {
    "United States of America": "US",
    "Russian Federation": "Russia",
}


def sci_fmt(x, _pos):
    return f"{x:.1e}"


def load_intensity() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(PROCESSED / "carbon intensity.xlsx")
    intensity = pd.DataFrame({"Year": raw["Year"]})
    co2 = pd.DataFrame({"Year": raw["Year"]})
    gdp = pd.DataFrame({"Year": raw["Year"]})
    for country in COUNTRIES:
        intensity[country] = raw[f"CO2/GDP-{country}"]
        co2[country] = raw[f"CO2 Emission/Mt-{country}"]
        gdp[country] = raw[f"GDP/US$-{country}"]
    return intensity, co2, gdp


def plot_carbon_intensity(intensity: pd.DataFrame, gdp: pd.DataFrame, outdir) -> None:
    set_plot_style()
    colors = {
        "China": "#C23B22",
        "World": "#4C78A8",
        "US": "#6C6C6C",
        "India": "#2A9D8F",
        "Russia": "#8D5A97",
        "Japan": "#7B8CDE",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for country in COUNTRIES:
        axes[0].plot(intensity["Year"], intensity[country], label=country, color=colors[country], linewidth=2 if country == "China" else 1.3)
    axes[0].set_ylabel("CO2/GDP")
    axes[0].set_xlabel("Year")
    axes[0].set_title("Carbon intensity")
    axes[0].yaxis.set_major_formatter(FuncFormatter(sci_fmt))
    axes[0].grid(axis="y", alpha=0.25)

    norm = intensity.copy()
    for country in COUNTRIES:
        norm[country] = norm[country] / norm[country].iloc[0]
        axes[1].plot(norm["Year"], norm[country], label=country, color=colors[country], linewidth=2 if country == "China" else 1.3)
    axes[1].set_ylabel("Normalized CO2/GDP")
    axes[1].set_xlabel("Year")
    axes[1].set_title("Normalized to 2000")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    savefig(fig, outdir, "fig4ab_global_carbon_intensity")

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    for country in COUNTRIES:
        ax.plot(gdp[country] / 1e12, intensity[country], label=country, color=colors[country], linewidth=2 if country == "China" else 1.3)
        ax.scatter(gdp[country].iloc[-1] / 1e12, intensity[country].iloc[-1], color=colors[country], s=22)
    ax.set_xlabel("GDP (trillion US$)")
    ax.set_ylabel("CO2/GDP")
    ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
    ax.set_title("GDP-carbon intensity trajectory")
    ax.legend(frameon=False)
    ax.grid(alpha=0.20)
    savefig(fig, outdir, "fig4c_gdp_carbon_intensity_trajectory")


def plot_policy_subjects(outdir) -> None:
    path = PROCESSED / "global_policy" / "selected_countries_and_world_primary_subject_ratio.xlsx"
    df = pd.read_excel(path)
    df["Country"] = df["Country"].replace(COUNTRY_NAME_MAP)
    subjects = ["Climate change", "Environment", "Energy", "Forestry", "Land and soil", "Water"]
    sub = df[df["Primary subject"].isin(subjects)].copy()
    pivot = sub.pivot_table(index=["Country", "Year"], columns="Primary subject", values="primary_subject_ratio", aggfunc="mean").fillna(0)
    pivot = pivot.reset_index()
    pivot.to_csv(outdir / "global_policy_subject_ratios.csv", index=False)

    set_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, country in zip(axes, COUNTRIES):
        cdf = pivot[pivot["Country"] == country].sort_values("Year")
        if cdf.empty:
            ax.axis("off")
            continue
        ax.stackplot(cdf["Year"], [cdf[s] if s in cdf else 0 for s in subjects], labels=subjects, alpha=0.85)
        ax.set_title(country)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.20)
    axes[0].legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.30))
    fig.suptitle("Environmental policy subject composition")
    savefig(fig, outdir, "fig4d_policy_subject_composition")


def main() -> None:
    outdir = ensure_output_dir("05_global_comparison")
    intensity, _co2, gdp = load_intensity()
    plot_carbon_intensity(intensity, gdp, outdir)
    plot_policy_subjects(outdir)
    print(f"Saved global comparison outputs to {outdir}")


if __name__ == "__main__":
    main()
