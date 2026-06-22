#!/usr/bin/env python
"""
Tabla unificada de alucinación por país y modelo.

Combina (solo lectura):
  - category_proportions_by_country.csv  → % alucinación + IC 95 % bootstrap
  - alucinacion_composition_residual.csv → esperado por mezcla + residual
  - calcula IR (misma fórmula que representativity_index.py)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.representativity_index import (  # noqa: E402
    compute_ir,
    interpret_ir,
)
from results_io import DEFAULT_OUTPUT_DIR  # noqa: E402

PROPORTIONS_FILE = "category_proportions_by_country.csv"
RESIDUAL_FILE = "alucinacion_composition_residual.csv"
OUTPUT_FILENAME = "alucinacion_country_full.csv"


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def build_alucinacion_country_full(
    proportions_df: pd.DataFrame,
    residual_df: pd.DataFrame,
) -> pd.DataFrame:
    prop_cols = [
        "pais",
        "model",
        "n_total",
        "n_alucinacion",
        "pct_alucinacion",
        "pct_alucinacion_ci_lower",
        "pct_alucinacion_ci_upper",
        "confiable",
    ]
    missing_prop = [c for c in prop_cols if c not in proportions_df.columns]
    if missing_prop:
        raise ValueError(f"{PROPORTIONS_FILE} missing columns: {missing_prop}")

    resid_cols = [
        "pais",
        "model",
        "pct_alucinacion_esperado_por_mezcla",
        "residual_alucinacion",
        "interpretacion",
    ]
    missing_resid = [c for c in resid_cols if c not in residual_df.columns]
    if missing_resid:
        raise ValueError(f"{RESIDUAL_FILE} missing columns: {missing_resid}")

    props = proportions_df[prop_cols].copy()
    resid = residual_df[resid_cols].rename(
        columns={"interpretacion": "interpretacion_composicion"}
    )

    merged = props.merge(resid, on=["pais", "model"], how="inner", validate="one_to_one")
    if len(merged) != len(props):
        raise ValueError("Merge pais×model did not match one-to-one between input CSVs.")

    merged["IR"] = merged.apply(
        lambda row: round(
            compute_ir(row["pct_alucinacion"], row["residual_alucinacion"]),
            2,
        ),
        axis=1,
    )
    merged["interpretacion_ir"] = merged["IR"].map(interpret_ir)

    merged["ic_95_alucinacion"] = merged.apply(
        lambda row: f"[{row['pct_alucinacion_ci_lower']:.1f}–{row['pct_alucinacion_ci_upper']:.1f}]",
        axis=1,
    )

    column_order = [
        "model",
        "pais",
        "n_total",
        "n_alucinacion",
        "pct_alucinacion",
        "pct_alucinacion_ci_lower",
        "pct_alucinacion_ci_upper",
        "ic_95_alucinacion",
        "pct_alucinacion_esperado_por_mezcla",
        "residual_alucinacion",
        "IR",
        "confiable",
        "interpretacion_composicion",
        "interpretacion_ir",
    ]
    return merged[column_order].sort_values(
        ["model", "pct_alucinacion", "residual_alucinacion"],
        ascending=[True, False, False],
        kind="stable",
    ).reset_index(drop=True)


def print_report_table(full_df: pd.DataFrame, top_n: int = 5) -> None:
    for model in sorted(full_df["model"].unique()):
        subset = full_df[full_df["model"] == model].head(top_n)
        print(f"\n=== {model} — top {top_n} por % alucinación ===")
        display = subset[
            [
                "pais",
                "pct_alucinacion",
                "ic_95_alucinacion",
                "residual_alucinacion",
                "IR",
                "interpretacion_ir",
            ]
        ].copy()
        display["pct_alucinacion"] = display["pct_alucinacion"].map(lambda x: f"{x:.1f}%")
        display["residual_alucinacion"] = display["residual_alucinacion"].map(
            lambda x: f"{x:+.1f}pp"
        )
        display["IR"] = display["IR"].map(lambda x: f"{x:.2f}")
        print(display.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge country hallucination rate, CI, residual and IR into one CSV."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory with input and output CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proportions_path = args.results_dir / PROPORTIONS_FILE
    residual_path = args.results_dir / RESIDUAL_FILE
    _require_file(proportions_path, PROPORTIONS_FILE)
    _require_file(residual_path, RESIDUAL_FILE)

    full_df = build_alucinacion_country_full(
        pd.read_csv(proportions_path),
        pd.read_csv(residual_path),
    )

    output_path = args.results_dir / OUTPUT_FILENAME
    full_df.to_csv(output_path, index=False, encoding="utf-8")

    print_report_table(full_df)
    print(f"\nSaved {len(full_df)} rows to {output_path.resolve()}")


if __name__ == "__main__":
    main()
