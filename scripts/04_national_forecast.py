from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import PROCESSED, ensure_output_dir, savefig, set_plot_style


def load_scenarios(folder: Path) -> dict[str, pd.DataFrame]:
    scenarios = {}
    for path in folder.glob("scenario_*_co2_path_no_regime.csv"):
        name = path.name.replace("scenario_", "").replace("_co2_path_no_regime.csv", "")
        scenarios[name] = pd.read_csv(path)
    return scenarios


def main() -> None:
    outdir = ensure_output_dir("04_national_forecast")
    folder = PROCESSED / "forecast" / "national_co2_elasticnet_dual_carbon_no_regime"
    scenarios = load_scenarios(folder)
    if not scenarios:
        raise FileNotFoundError(f"No scenario files found in {folder}")

    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    order = ["Baseline", "100%_C0_share", "100%_C1_share", "100%_C2_share", "100%_C3_share", "100%_C4_share", "100%_C5_share", "100%_C6_share"]
    for name in order:
        if name not in scenarios:
            continue
        df = scenarios[name]
        lw = 2.4 if name == "Baseline" else 1.1
        alpha = 1.0 if name == "Baseline" else 0.70
        label = name.replace("_", " ")
        ax.plot(df["year"], df["yhat"], linewidth=lw, alpha=alpha, label=label)
    ax.axvline(2030, color="0.45", linestyle="--", linewidth=0.8)
    ax.axvline(2060, color="0.45", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Predicted CO2 emissions")
    ax.set_title("National CO2 forecasting scenarios")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    savefig(fig, outdir, "fig5_national_co2_forecast")

    for name, df in scenarios.items():
        df.to_csv(outdir / f"{name}_scenario.csv", index=False)
    metrics = folder / "elasticnet_no_regime_metrics.csv"
    coeffs = folder / "elasticnet_coefficients_unscaled_no_regime.csv"
    if metrics.exists():
        pd.read_csv(metrics).to_csv(outdir / "model_metrics.csv", index=False)
    if coeffs.exists():
        pd.read_csv(coeffs).to_csv(outdir / "model_coefficients.csv", index=False)
    print(f"Saved national forecast outputs to {outdir}")


if __name__ == "__main__":
    main()
