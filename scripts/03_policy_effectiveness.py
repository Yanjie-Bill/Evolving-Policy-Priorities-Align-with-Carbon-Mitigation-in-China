from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common import PROCESSED, ensure_output_dir, savefig, set_plot_style


NEV_POLICY = "国务院办公厅关于印发新能源汽车产业发展规划（2021—2035年）的通知"


def main() -> None:
    outdir = ensure_output_dir("03_policy_effectiveness")
    monthly = pd.read_csv(PROCESSED / "did" / "monthly_DID_policy_impact_results.csv")
    annual = pd.read_csv(PROCESSED / "did" / "DID_policy_impact_results.csv")
    monthly["is_nev"] = monthly["policy"].str.contains("新能源汽车产业发展规划", na=False)
    annual["is_nev"] = annual["policy"].str.contains("新能源汽车产业发展规划", na=False)

    monthly.sort_values(["pvalue", "coef"]).to_csv(outdir / "monthly_did_ranked.csv", index=False)
    annual.sort_values(["pvalue", "coef"]).to_csv(outdir / "annual_did_ranked.csv", index=False)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    colors = monthly["is_nev"].map({True: "#C23B22", False: "#8A8F98"})
    ax.scatter(monthly["coef"], -monthly["pvalue"].apply(lambda p: __import__("math").log10(max(p, 1e-12))),
               c=colors, s=45, alpha=0.85)
    ax.axvline(0, color="0.4", linestyle="--", linewidth=0.8)
    ax.axhline(-__import__("math").log10(0.05), color="0.4", linestyle=":", linewidth=0.8)
    ax.set_xlabel("DID coefficient")
    ax.set_ylabel("-log10(p value)")
    ax.set_title("Policy-effect screening by monthly DID")
    for _, row in monthly[monthly["is_nev"]].iterrows():
        ax.annotate("NEV plan", (row["coef"], -__import__("math").log10(max(row["pvalue"], 1e-12))),
                    xytext=(5, 5), textcoords="offset points", fontsize=8, color="#C23B22")
    savefig(fig, outdir, "fig3a_did_policy_screening")

    nev = monthly[monthly["is_nev"]]
    with open(outdir / "nev_did_result.txt", "w", encoding="utf-8") as f:
        if nev.empty:
            f.write("No NEV policy row found in monthly_DID_policy_impact_results.csv\n")
        else:
            f.write(nev.to_string(index=False))
            f.write("\n")
    print(f"Saved DID outputs to {outdir}")


if __name__ == "__main__":
    main()
