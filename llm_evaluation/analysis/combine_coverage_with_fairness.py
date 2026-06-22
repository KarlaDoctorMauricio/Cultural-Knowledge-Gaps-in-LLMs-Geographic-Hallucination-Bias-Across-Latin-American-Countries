#!/usr/bin/env python
"""
Merge thematic coverage (UMAP dispersion) with fairness results by country.

Reads existing CSV outputs only; does not run evaluation or coverage pipelines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_COVERAGE_DIR = ROOT / "data" / "coverage_sample"
DEFAULT_RESULTS_DIR = ROOT / "data" / "results"
DEFAULT_MODEL = "GPT"


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}. Run the upstream pipeline first."
        )


def load_coverage_table(coverage_dir: Path) -> pd.DataFrame:
    path = coverage_dir / "coverage_by_country.csv"
    _require_file(path, "coverage_by_country.csv")
    df = pd.read_csv(path)
    required = {"pais", "dispersion_promedio", "n_preguntas"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    return df[list(required)].rename(columns={"n_preguntas": "n_preguntas_coverage"})


def load_fairness_by_country(results_dir: Path, model: str) -> pd.DataFrame:
    proportions_path = results_dir / "category_proportions_by_country.csv"
    mae_path = results_dir / "group_mae_pais.csv"

    _require_file(proportions_path, "category_proportions_by_country.csv")

    proportions = pd.read_csv(proportions_path)
    proportions = proportions[proportions["model"] == model].copy()
    if proportions.empty:
        raise ValueError(f"No rows for model={model} in {proportions_path.name}")

    prop_cols = ["pais", "pct_alucinacion", "n_total"]
    missing = [col for col in prop_cols if col not in proportions.columns]
    if missing:
        raise ValueError(f"{proportions_path.name} missing columns: {missing}")

    merged = proportions[prop_cols].rename(columns={"n_total": "n_preguntas_judged"})

    if mae_path.exists():
        mae = pd.read_csv(mae_path)
        mae = mae[mae["method"] == model][["pais", "mae", "n_preguntas"]].copy()
        mae = mae.rename(columns={"n_preguntas": "n_preguntas_mae"})
        merged = merged.merge(mae, on="pais", how="left")
    else:
        merged["mae"] = np.nan
        merged["n_preguntas_mae"] = np.nan

    return merged


def build_coverage_vs_fairness_table(
    coverage_dir: Path,
    results_dir: Path,
    model: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    coverage = load_coverage_table(coverage_dir)
    fairness = load_fairness_by_country(results_dir, model=model)

    table = coverage.merge(fairness, on="pais", how="inner")
    table.insert(0, "model", model)
    table["n_preguntas"] = table["n_preguntas_coverage"]

    return table[
        [
            "model",
            "pais",
            "dispersion_promedio",
            "pct_alucinacion",
            "mae",
            "n_preguntas",
            "n_preguntas_judged",
        ]
    ].sort_values("dispersion_promedio", ascending=False)


def compute_correlations(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return Spearman and Pearson rho/p-value for dispersion vs fairness metrics."""
    try:
        from scipy.stats import pearsonr, spearmanr
    except ImportError as exc:
        raise ImportError(
            "scipy is required for correlation p-values. Install with: pip install scipy"
        ) from exc

    results: dict[str, dict[str, float]] = {}
    pairs = [
        ("pct_alucinacion", "dispersion vs pct_alucinacion"),
        ("mae", "dispersion vs mae"),
    ]

    for column, label in pairs:
        if column not in table.columns:
            continue
        subset = table[["dispersion_promedio", column]].dropna()
        if len(subset) < 3:
            results[label] = {
                "spearman_r": np.nan,
                "spearman_p": np.nan,
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "n": float(len(subset)),
            }
            continue

        x = subset["dispersion_promedio"].to_numpy()
        y = subset[column].to_numpy()
        sp_r, sp_p = spearmanr(x, y)
        pe_r, pe_p = pearsonr(x, y)
        results[label] = {
            "spearman_r": float(sp_r),
            "spearman_p": float(sp_p),
            "pearson_r": float(pe_r),
            "pearson_p": float(pe_p),
            "n": float(len(subset)),
        }

    return results


def plot_dispersion_vs_fairness(
    table: pd.DataFrame,
    output_path: Path,
    *,
    y_column: str = "pct_alucinacion",
    model: str = DEFAULT_MODEL,
) -> Path:
    subset = table[["pais", "dispersion_promedio", y_column]].dropna()
    if subset.empty:
        raise ValueError(f"No data to plot for y_column={y_column}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        subset["dispersion_promedio"],
        subset[y_column],
        s=60,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.5,
    )

    for _, row in subset.iterrows():
        ax.annotate(
            row["pais"],
            (row["dispersion_promedio"], row[y_column]),
            fontsize=7,
            alpha=0.85,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("dispersion_promedio (cobertura tematica UMAP)")
    ax.set_ylabel(y_column)
    ax.set_title(f"Cobertura vs fairness por pais ({model})")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_correlation_report(correlations: dict[str, dict[str, float]]) -> None:
    print("\n--- Correlacion cobertura tematica vs fairness (por pais) ---")
    for label, stats in correlations.items():
        print(f"\n{label} (n={int(stats['n'])}):")
        print(
            f"  Spearman rho={stats['spearman_r']:.3f}, p={stats['spearman_p']:.4f} "
            "(recomendado con n~18 paises)"
        )
        print(f"  Pearson  r={stats['pearson_r']:.3f}, p={stats['pearson_p']:.4f}")


def run_combine_analysis(
    coverage_dir: Path,
    results_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    output_csv: Optional[Path] = None,
    output_plot: Optional[Path] = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], Path, Path]:
    table = build_coverage_vs_fairness_table(coverage_dir, results_dir, model=model)
    correlations = compute_correlations(table)

    csv_path = output_csv or (coverage_dir / "coverage_vs_fairness.csv")
    plot_path = output_plot or (coverage_dir / "plots" / "dispersion_vs_fairness.png")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False, encoding="utf-8")
    plot_dispersion_vs_fairness(table, plot_path, y_column="pct_alucinacion", model=model)

    return table, correlations, csv_path, plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine coverage dispersion with fairness metrics by country."
    )
    parser.add_argument(
        "--coverage-dir",
        type=Path,
        default=DEFAULT_COVERAGE_DIR,
        help=f"Directory with coverage_by_country.csv (default: {DEFAULT_COVERAGE_DIR}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory with fairness CSVs (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to merge (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <coverage-dir>/coverage_vs_fairness.csv).",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help=(
            "Output plot path "
            "(default: <coverage-dir>/plots/dispersion_vs_fairness.png)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table, correlations, csv_path, plot_path = run_combine_analysis(
        args.coverage_dir,
        args.results_dir,
        model=args.model,
        output_csv=args.output_csv,
        output_plot=args.output_plot,
    )
    print_correlation_report(correlations)
    print(f"\nSaved merged table to {csv_path.resolve()}")
    print(f"Saved scatter plot to {plot_path.resolve()}")
    print(f"Rows merged: {len(table)}")


if __name__ == "__main__":
    main()
