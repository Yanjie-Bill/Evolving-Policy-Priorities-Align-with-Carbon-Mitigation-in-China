from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"


CLUSTER_LABELS = {
    0: "C0 Electrification",
    1: "C1 Green buildings",
    2: "C2 Ecology",
    3: "C3 Energy efficiency",
    4: "C4 Industry",
    5: "C5 Governance",
    6: "C6 Coal and mining",
}


CLUSTER_COLORS = {
    0: "#4C78A8",
    1: "#F58518",
    2: "#54A24B",
    3: "#E45756",
    4: "#72B7B2",
    5: "#B279A2",
    6: "#9D755D",
}


def ensure_output_dir(name: str) -> Path:
    path = OUTPUTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def savefig(fig, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
