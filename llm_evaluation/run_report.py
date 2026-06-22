#!/usr/bin/env python
"""
Genera todas las tablas del informe a partir de CSVs existentes.

Un solo comando tras ``run_all.py`` y (opcional) ``rerun_judge_stability.py``:

    python run_report.py

También se invoca al final de ``run_all.py --report`` (por defecto activo).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.alucinacion_country_full import (  # noqa: E402
    OUTPUT_FILENAME as COUNTRY_FULL_FILENAME,
    build_alucinacion_country_full,
)
from analysis.analyze_alucinacion_composition import (  # noqa: E402
    load_results_from_dir,
    print_alucinacion_composition_summary,
    resolve_classifications_path,
    run_alucinacion_composition_analysis,
)
from analysis.category_proportions_by_country import (  # noqa: E402
    print_fairness_summary,
    run_category_proportions_analysis,
)
from analysis.combine_coverage_with_fairness import (  # noqa: E402
    run_combine_analysis,
)
from analysis.representativity_index import (  # noqa: E402
    OUTPUT_FILENAME as IR_FILENAME,
    SUMMARY_FILENAME as IR_SUMMARY_FILENAME,
    build_representativity_index,
    build_summary_table,
    print_index_table,
)
from fairness_toolkit.choclo import LOCAL_SAMPLE_PATH  # noqa: E402
from results_io import DEFAULT_OUTPUT_DIR  # noqa: E402

DEFAULT_COVERAGE_DIR = ROOT / "data" / "coverage_sample"
RESIDUAL_FILENAME = "alucinacion_composition_residual.csv"
PROPORTIONS_FILENAME = "category_proportions_by_country.csv"


def run_coverage_analysis(
    *,
    coverage_dir: Path,
    random_state: int = 42,
) -> None:
    """Run UMAP coverage pipeline (embeddings + plots)."""
    cmd = [
        sys.executable,
        str(ROOT / "analysis" / "analyze_coverage.py"),
        "--mode",
        "sample",
        "--output-dir",
        str(coverage_dir),
        "--random-state",
        str(random_state),
    ]
    print(f"\n[report] UMAP cobertura: {' '.join(cmd[2:])}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_report_analyses(
    results_dir: Path = DEFAULT_OUTPUT_DIR,
    coverage_dir: Path = DEFAULT_COVERAGE_DIR,
    *,
    refresh_judge_tables: bool = True,
    run_coverage: bool = True,
    combine_models: Sequence[str] = ("GPT", "Claude"),
    min_n: int = 5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    save_proportion_chart: bool = True,
    classifications: pd.DataFrame | None = None,
    proportions_df: pd.DataFrame | None = None,
    scored_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """
    Build all report CSVs/plots from ``results_dir``.

    If ``refresh_judge_tables`` is True (default for standalone ``run_report.py``),
    recomputes proportions and composition from ``judge_final_results.csv``.

    If False (called from ``run_all.py`` after step 5), expects those CSVs
    to exist from the current evaluation run.
    """
    results_dir = Path(results_dir)
    coverage_dir = Path(coverage_dir)
    saved: dict[str, Path] = {}

    if refresh_judge_tables:
        print("[report] Proporciones del judge + residual composicional ...")
        loaded = load_results_from_dir(results_dir)
        classified = classifications if classifications is not None else loaded["classified"]
        if isinstance(classified, Path):
            classified = pd.read_csv(classified)
        props_input = proportions_df if proportions_df is not None else loaded.get("proportions")
        scored = scored_df if scored_df is not None else loaded.get("scored_df")

        if scored is None:
            raise FileNotFoundError(
                f"Missing evaluation_results.csv in {results_dir}. Run run_all.py first."
            )

        prop_saved = run_category_proportions_analysis(
            classified,
            results_dir,
            min_n=min_n,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            save_chart=save_proportion_chart,
        )
        print_fairness_summary(prop_saved["fairness_gaps_df"])
        for key, value in prop_saved.items():
            if isinstance(value, Path):
                saved[key] = value

        comp_saved = run_alucinacion_composition_analysis(
            classified,
            prop_saved["proportions_df"],
            scored,
            results_dir,
        )
        print_alucinacion_composition_summary(comp_saved)
        for name, df in comp_saved.items():
            if df is not None and not df.empty:
                saved[name] = results_dir / name
    else:
        print("[report] Usando proporciones y residual ya generados en esta corrida ...")
        for name in (PROPORTIONS_FILENAME, RESIDUAL_FILENAME):
            path = results_dir / name
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path.name}. Run run_all.py first or use refresh_judge_tables."
                )

    print("[report] Tabla unificada alucinacion_country_full.csv ...")
    proportions = pd.read_csv(results_dir / PROPORTIONS_FILENAME)
    residual = pd.read_csv(results_dir / RESIDUAL_FILENAME)
    country_full = build_alucinacion_country_full(proportions, residual)
    country_full_path = results_dir / COUNTRY_FULL_FILENAME
    country_full.to_csv(country_full_path, index=False, encoding="utf-8")
    saved[COUNTRY_FULL_FILENAME] = country_full_path
    print(f"      {len(country_full)} filas -> {country_full_path.name}")

    print("[report] Índice de Representatividad (IR) ...")
    index_df = build_representativity_index(residual)
    summary_df = build_summary_table(index_df)
    index_path = results_dir / IR_FILENAME
    summary_path = results_dir / IR_SUMMARY_FILENAME
    index_df.to_csv(index_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    saved[IR_FILENAME] = index_path
    saved[IR_SUMMARY_FILENAME] = summary_path
    print_index_table(index_df)

    if run_coverage:
        if not LOCAL_SAMPLE_PATH.exists():
            print(f"[report] Skip UMAP: missing {LOCAL_SAMPLE_PATH}")
        else:
            run_coverage_analysis(coverage_dir=coverage_dir, random_state=random_state)
            saved["coverage_by_country.csv"] = coverage_dir / "coverage_by_country.csv"

            for model in combine_models:
                print(f"[report] Cobertura vs fairness ({model}) ...")
                csv_name = (
                    "coverage_vs_fairness.csv"
                    if model == "Claude"
                    else f"coverage_vs_fairness_{model.lower()}.csv"
                )
                plot_name = (
                    "dispersion_vs_fairness.png"
                    if model == "Claude"
                    else f"dispersion_vs_fairness_{model.lower()}.png"
                )
                _, _, csv_path, plot_path = run_combine_analysis(
                    coverage_dir,
                    results_dir,
                    model=model,
                    output_csv=coverage_dir / csv_name,
                    output_plot=coverage_dir / "plots" / plot_name,
                )
                saved[csv_name] = csv_path
                saved[plot_name] = plot_path

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all report tables (proportions, IR, country full, UMAP)."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Results directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--coverage-dir",
        type=Path,
        default=DEFAULT_COVERAGE_DIR,
        help=f"UMAP output directory (default: {DEFAULT_COVERAGE_DIR}).",
    )
    parser.add_argument(
        "--skip-coverage",
        action="store_true",
        help="Skip UMAP coverage and coverage_vs_fairness (faster).",
    )
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help=(
            "Do not recompute proportions/residual from judge CSV "
            "(use files already on disk)."
        ),
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=5,
        help="Minimum n per group for confiable flag.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples for IC.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for bootstrap and UMAP.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate judge source when refreshing
    if not args.skip_refresh:
        judge_path = resolve_classifications_path(args.results_dir)
        print(f"Using judge labels: {judge_path.resolve()}")

    print("=" * 60)
    print("CHOCLO report pipeline (tablas del informe)")
    print("=" * 60)

    saved = run_report_analyses(
        args.results_dir,
        args.coverage_dir,
        refresh_judge_tables=not args.skip_refresh,
        run_coverage=not args.skip_coverage,
        min_n=args.min_n,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
    )

    print("\n" + "=" * 60)
    print("Report done.")
    print(f"Outputs in: {args.results_dir.resolve()}")
    if not args.skip_coverage:
        print(f"Coverage:   {args.coverage_dir.resolve()}")
    for name in sorted(saved):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
