from __future__ import annotations

import os
import runpy
from pathlib import Path


SCRIPTS = [
    "01_national_policy_emission.py",
    "02_provincial_typology.py",
    "03_policy_effectiveness.py",
    "04_national_forecast.py",
    "05_global_comparison.py",
]


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent
    mpl_cache = root / "outputs" / ".matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    for script in SCRIPTS:
        print(f"\n=== Running {script} ===")
        runpy.run_path(str(here / script), run_name="__main__")
    print("\nAll reproducibility scripts completed.")


if __name__ == "__main__":
    main()
