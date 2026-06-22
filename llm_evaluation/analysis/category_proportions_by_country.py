#!/usr/bin/env python
"""
Primary fairness analysis: judge label proportions by country and model.

Uses ``low_score_classifications.csv`` (or equivalent) to compute per-country
quality breakdowns, bootstrap CIs for all judge label proportions, and
cross-country fairness gaps. Hallucination rate gap is the main fairness metric
for reporting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairness_toolkit.group_mae_stats import (  # noqa: E402
    INSUFFICIENT_SAMPLE_NOTE,
    _bootstrap_proportion,
)
from results_io import DEFAULT_OUTPUT_DIR  # noqa: E402

QUALITY_LABELS = ("correcta", "parcial", "alucinacion", "abstencion")
DEFAULT_JUDGE_MODELS = ("GPT", "Claude")
PRIMARY_FAIRNESS_METRIC = "bias_alucinacion"
SECONDARY_FAIRNESS_METRICS = ("bias_correcta", "bias_abstencion")

LABEL_COLORS = {
    "correcta": "#2ca02c",
    "parcial": "#ffbf00",
    "alucinacion": "#d62728",
    "abstencion": "#7f7f7f",
}

PROPORTION_OUTPUT_FILES = (
    "category_proportions_by_country.csv",
    "category_proportions_global_summary.csv",
    "category_proportions_fairness_gaps.csv",
    "category_proportions_chart.png",
)


def _proportion_ci_column_names(label: str) -> tuple[str, str]:
    return f"pct_{label}_ci_lower", f"pct_{label}_ci_upper"


def proportion_table_columns(*, include_ci: bool = True) -> list[str]:
    columns = [
        "pais",
        "model",
        "n_total",
        *[f"n_{label}" for label in QUALITY_LABELS],
        *[f"pct_{label}" for label in QUALITY_LABELS],
    ]
    if include_ci:
        for label in QUALITY_LABELS:
            columns.extend(_proportion_ci_column_names(label))
    columns.extend(["confiable", "nota"])
    return columns


def clear_stale_proportion_outputs(output_dir: Path) -> list[Path]:
    """Remove proportion outputs from a previous run so they cannot be misread."""
    removed: list[Path] = []
    for name in PROPORTION_OUTPUT_FILES:
        path = output_dir / name
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def verify_classifications_csv_fresh(
    path: Path,
    classifications: pd.DataFrame,
    run_started_at: float,
) -> None:
    """
    Ensure ``low_score_classifications.csv`` was written during the current run.

    ``run_all.py`` passes classifications in memory; this check guards against
    accidental disk reads or partial writes leaving stale artifacts.
    """
    if not path.exists():
        raise RuntimeError(
            f"Missing {path.name} after save_results; cannot confirm fresh classifications."
        )

    mtime = path.stat().st_mtime
    if mtime < run_started_at - 1.0:
        raise RuntimeError(
            f"Stale {path.name} on disk (modified before this run started). "
            "Delete it or re-run evaluation before computing proportions."
        )

    on_disk = pd.read_csv(path)
    if len(on_disk) != len(classifications):
        raise RuntimeError(
            f"{path.name} row count mismatch: disk={len(on_disk)}, "
            f"in-memory={len(classifications)}."
        )


def run_proportions_for_current_run(
    classifications: pd.DataFrame,
    output_dir: Path,
    *,
    classifications_csv: Path,
    run_started_at: float,
    model_names: Sequence[str] = DEFAULT_JUDGE_MODELS,
    min_n: int = 5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    save_chart: bool = True,
) -> dict[str, Path | pd.DataFrame]:
    """
    Run proportion analysis for ``run_all.py`` using in-memory classifications.

    Verifies the CSV saved in the same run matches before writing new outputs.
    Does **not** read classifications from disk for computation.
    """
    verify_classifications_csv_fresh(classifications_csv, classifications, run_started_at)
    clear_stale_proportion_outputs(output_dir)
    return run_category_proportions_analysis(
        classifications,
        output_dir,
        model_names=model_names,
        min_n=min_n,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        save_chart=save_chart,
    )


def _aggregate_proportion_row(
    labels: np.ndarray,
    *,
    pais: str,
    model: str,
    min_n: int,
    n_bootstrap: int,
    random_state: int,
) -> dict:
    n_total = int(len(labels))
    row: dict = {"pais": pais, "model": model, "n_total": n_total}

    for label in QUALITY_LABELS:
        count = int(np.sum(labels == label))
        row[f"n_{label}"] = count
        row[f"pct_{label}"] = round(100.0 * count / n_total, 2) if n_total else np.nan

        ci_lower, ci_upper, _ = _bootstrap_proportion(
            labels,
            label,
            n_bootstrap=n_bootstrap,
            random_state=random_state + hash(f"{pais}_{model}_{label}") % 10_000,
        )
        lower_col, upper_col = _proportion_ci_column_names(label)
        row[lower_col] = round(ci_lower, 2) if pd.notna(ci_lower) else np.nan
        row[upper_col] = round(ci_upper, 2) if pd.notna(ci_upper) else np.nan

    confiable = n_total >= min_n
    row["confiable"] = confiable
    row["nota"] = None if confiable else INSUFFICIENT_SAMPLE_NOTE
    return row


def _aggregate_country_model_row(
    subset: pd.DataFrame,
    *,
    pais: str,
    model: str,
    min_n: int,
    n_bootstrap: int,
    random_state: int,
) -> dict:
    labels = subset["judge_label"].astype(str).to_numpy()
    return _aggregate_proportion_row(
        labels,
        pais=pais,
        model=model,
        min_n=min_n,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )


def compute_category_proportions_by_country(
    classified: pd.DataFrame,
    model_names: Sequence[str] = DEFAULT_JUDGE_MODELS,
    min_n: int = 5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build per-country judge label proportions for each model."""
    columns = proportion_table_columns(include_ci=True)

    if classified.empty or "judge_label" not in classified.columns:
        return pd.DataFrame(columns=columns)

    required = {"pais", "model", "judge_label"}
    missing = required - set(classified.columns)
    if missing:
        raise ValueError(f"Classifications missing columns: {sorted(missing)}")

    filtered = classified[classified["model"].isin(model_names)].copy()
    if filtered.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    for (pais, model), subset in filtered.groupby(["pais", "model"], sort=True):
        rows.append(
            _aggregate_country_model_row(
                subset,
                pais=str(pais),
                model=str(model),
                min_n=min_n,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
        )

    result = pd.DataFrame(rows)
    return result.sort_values(["model", "pct_alucinacion"], ascending=[True, False]).reset_index(
        drop=True
    )


def compute_global_proportion_summary(
    classified: pd.DataFrame,
    model_names: Sequence[str] = DEFAULT_JUDGE_MODELS,
    min_n: int = 5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Aggregate judge proportions by model (all countries) with bootstrap CIs."""
    columns = proportion_table_columns(include_ci=True)
    if classified.empty or "judge_label" not in classified.columns:
        return pd.DataFrame(columns=columns)

    filtered = classified[classified["model"].isin(model_names)].copy()
    if filtered.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    for model, subset in filtered.groupby("model", sort=True):
        labels = subset["judge_label"].astype(str).to_numpy()
        rows.append(
            _aggregate_proportion_row(
                labels,
                pais="__global__",
                model=str(model),
                min_n=min_n,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
        )

    return pd.DataFrame(rows).reset_index(drop=True)


def compute_fairness_gaps(
    proportions: pd.DataFrame,
    metric_prefixes: Sequence[str] = ("alucinacion", "correcta", "abstencion"),
) -> pd.DataFrame:
    """
    Compute max-min country gaps for selected proportion metrics, per model.

    ``bias_alucinacion`` is the primary fairness gap for reporting.
    """
    rows: list[dict] = []

    for model, subset in proportions.groupby("model", sort=True):
        row: dict = {"model": model}
        for prefix in metric_prefixes:
            col = f"pct_{prefix}"
            if col not in subset.columns or subset[col].dropna().empty:
                row[f"bias_{prefix}"] = np.nan
                row[f"max_pais_{prefix}"] = None
                row[f"min_pais_{prefix}"] = None
                row[f"max_pct_{prefix}"] = np.nan
                row[f"min_pct_{prefix}"] = np.nan
                continue

            valid = subset.dropna(subset=[col])
            max_idx = valid[col].idxmax()
            min_idx = valid[col].idxmin()
            max_pct = float(valid.loc[max_idx, col])
            min_pct = float(valid.loc[min_idx, col])

            row[f"bias_{prefix}"] = round(max_pct - min_pct, 2)
            row[f"max_pais_{prefix}"] = valid.loc[max_idx, "pais"]
            row[f"min_pais_{prefix}"] = valid.loc[min_idx, "pais"]
            row[f"max_pct_{prefix}"] = max_pct
            row[f"min_pct_{prefix}"] = min_pct

        rows.append(row)

    return pd.DataFrame(rows)


def plot_category_proportions_chart(
    proportions: pd.DataFrame,
    output_path: Path,
    model_names: Optional[Sequence[str]] = None,
) -> Path:
    """Stacked bar chart of judge label proportions by country, one panel per model."""
    if proportions.empty:
        raise ValueError("Cannot plot empty proportions table.")

    models = list(model_names or sorted(proportions["model"].unique()))
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 8), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        subset = proportions[proportions["model"] == model].sort_values(
            "pct_alucinacion", ascending=False
        )
        countries = subset["pais"].tolist()
        x = np.arange(len(countries))
        bottom = np.zeros(len(countries))

        for label in QUALITY_LABELS:
            values = subset[f"pct_{label}"].fillna(0.0).to_numpy()
            ax.bar(
                x,
                values,
                bottom=bottom,
                label=label,
                color=LABEL_COLORS[label],
                edgecolor="white",
                linewidth=0.5,
            )
            bottom += values

        ax.set_title(f"{model}")
        ax.set_xlabel("Pais")
        ax.set_ylabel("% respuestas")
        ax.set_xticks(x)
        ax.set_xticklabels(countries, rotation=60, ha="right")
        ax.set_ylim(0, 100)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(QUALITY_LABELS), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Distribucion de calidad por pais (judge)", y=1.06, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_category_proportions_analysis(
    classified: pd.DataFrame,
    output_dir: Path,
    model_names: Sequence[str] = DEFAULT_JUDGE_MODELS,
    min_n: int = 5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    save_chart: bool = True,
) -> dict[str, Path | pd.DataFrame]:
    """Compute proportions, fairness gaps, and optional chart."""
    proportions = compute_category_proportions_by_country(
        classified,
        model_names=model_names,
        min_n=min_n,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    global_summary = compute_global_proportion_summary(
        classified,
        model_names=model_names,
        min_n=min_n,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    gaps = compute_fairness_gaps(proportions)

    output_dir.mkdir(parents=True, exist_ok=True)
    proportions_path = output_dir / "category_proportions_by_country.csv"
    global_summary_path = output_dir / "category_proportions_global_summary.csv"
    gaps_path = output_dir / "category_proportions_fairness_gaps.csv"
    proportions.to_csv(proportions_path, index=False, encoding="utf-8")
    global_summary.to_csv(global_summary_path, index=False, encoding="utf-8")
    gaps.to_csv(gaps_path, index=False, encoding="utf-8")

    saved: dict[str, Path | pd.DataFrame] = {
        "category_proportions_by_country.csv": proportions_path,
        "category_proportions_global_summary.csv": global_summary_path,
        "category_proportions_fairness_gaps.csv": gaps_path,
        "proportions_df": proportions,
        "global_summary_df": global_summary,
        "fairness_gaps_df": gaps,
    }

    if save_chart and not proportions.empty:
        chart_path = output_dir / "category_proportions_chart.png"
        plot_category_proportions_chart(proportions, chart_path, model_names=model_names)
        saved["category_proportions_chart.png"] = chart_path

    return saved


def print_fairness_summary(gaps: pd.DataFrame) -> None:
    if gaps.empty:
        return

    print("\n--- Fairness principal: gap de alucinacion por pais (max - min) ---")
    for _, row in gaps.iterrows():
        print(
            f"  {row['model']}: bias_alucinacion={row.get('bias_alucinacion', np.nan):.2f} pp "
            f"(max {row.get('max_pais_alucinacion')} {row.get('max_pct_alucinacion', np.nan):.1f}%, "
            f"min {row.get('min_pais_alucinacion')} {row.get('min_pct_alucinacion', np.nan):.1f}%)"
        )
        print(
            f"           bias_correcta={row.get('bias_correcta', np.nan):.2f} pp, "
            f"bias_abstencion={row.get('bias_abstencion', np.nan):.2f} pp"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute judge label proportions by country (primary fairness metric)."
    )
    parser.add_argument(
        "--classifications",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "low_score_classifications.csv",
        help="Path to judge classifications CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSV and chart.",
    )
    parser.add_argument("--min-n", type=int, default=5, help="Minimum n for confiable=True.")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--no-chart", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classified = pd.read_csv(args.classifications)
    saved = run_category_proportions_analysis(
        classified,
        args.output_dir,
        min_n=args.min_n,
        n_bootstrap=args.n_bootstrap,
        save_chart=not args.no_chart,
    )
    print_fairness_summary(saved["fairness_gaps_df"])
    print(f"\nSaved outputs to {args.output_dir.resolve()}")
    for name in sorted(k for k in saved if k.endswith(".csv") or k.endswith(".png")):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
