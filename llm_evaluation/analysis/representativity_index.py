#!/usr/bin/env python
"""
Índice de Representatividad (IR) por país y modelo.

Análisis de solo lectura sobre ``alucinacion_composition_residual.csv``.
Combina la tasa observada de alucinación con el residual composicional
(solo penaliza residuales positivos).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from results_io import DEFAULT_OUTPUT_DIR  # noqa: E402

DEFAULT_INPUT_FILE = "alucinacion_composition_residual.csv"
OUTPUT_FILENAME = "representativity_index.csv"
SUMMARY_FILENAME = "representativity_index_summary.csv"

REQUIRED_COLUMNS = (
    "pais",
    "model",
    "pct_alucinacion_observado",
    "residual_alucinacion",
)


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}. Run analyze_alucinacion_composition.py first."
        )


def compute_ir(pct_alucinacion: float, residual_alucinacion: float) -> float:
    """IR = pct × (1 + max(0, residual / 100))."""
    penalty = max(0.0, float(residual_alucinacion) / 100.0)
    return float(pct_alucinacion) * (1.0 + penalty)


def interpret_ir(ir: float) -> str:
    if ir > 50:
        return "riesgo alto: alucinación elevada no explicada por composición"
    if ir >= 25:
        return "riesgo moderado"
    return "riesgo bajo"


def build_representativity_index(residual_df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in residual_df.columns]
    if missing:
        raise ValueError(f"{DEFAULT_INPUT_FILE} missing columns: {missing}")

    working = residual_df[list(REQUIRED_COLUMNS)].copy()
    working["IR"] = working.apply(
        lambda row: round(
            compute_ir(
                row["pct_alucinacion_observado"],
                row["residual_alucinacion"],
            ),
            2,
        ),
        axis=1,
    )
    working["interpretacion"] = working["IR"].map(interpret_ir)

    result = working.rename(
        columns={"pct_alucinacion_observado": "pct_alucinacion"}
    )[
        [
            "pais",
            "model",
            "pct_alucinacion",
            "residual_alucinacion",
            "IR",
            "interpretacion",
        ]
    ]
    return result.sort_values(["model", "IR"], ascending=[True, False]).reset_index(
        drop=True
    )


def build_summary_table(index_df: pd.DataFrame, top_n: int = 5, bottom_n: int = 3) -> pd.DataFrame:
    rows: list[dict] = []

    for model, subset in index_df.groupby("model", sort=True):
        ordered = subset.sort_values("IR", ascending=False, kind="stable")

        for rank, (_, row) in enumerate(ordered.head(top_n).iterrows(), start=1):
            rows.append(
                {
                    "model": model,
                    "grupo": f"top_{top_n}",
                    "rank": rank,
                    "pais": row["pais"],
                    "pct_alucinacion": row["pct_alucinacion"],
                    "residual_alucinacion": row["residual_alucinacion"],
                    "IR": row["IR"],
                    "interpretacion": row["interpretacion"],
                }
            )

        bottom = ordered.tail(bottom_n).sort_values("IR", ascending=True, kind="stable")
        for rank, (_, row) in enumerate(bottom.iterrows(), start=1):
            rows.append(
                {
                    "model": model,
                    "grupo": f"bottom_{bottom_n}",
                    "rank": rank,
                    "pais": row["pais"],
                    "pct_alucinacion": row["pct_alucinacion"],
                    "residual_alucinacion": row["residual_alucinacion"],
                    "IR": row["IR"],
                    "interpretacion": row["interpretacion"],
                }
            )

    summary = pd.DataFrame(rows)
    return summary[
        [
            "model",
            "grupo",
            "rank",
            "pais",
            "pct_alucinacion",
            "residual_alucinacion",
            "IR",
            "interpretacion",
        ]
    ]


def print_index_table(index_df: pd.DataFrame) -> None:
    display = index_df.copy()
    display["pct_alucinacion"] = display["pct_alucinacion"].map(lambda x: f"{x:.2f}")
    display["residual_alucinacion"] = display["residual_alucinacion"].map(
        lambda x: f"{x:+.2f}"
    )
    display["IR"] = display["IR"].map(lambda x: f"{x:.2f}")

    for model in sorted(display["model"].unique()):
        subset = display[display["model"] == model]
        print(f"\n=== {model} — Índice de Representatividad (IR descendente) ===")
        print(
            subset[["pais", "pct_alucinacion", "residual_alucinacion", "IR", "interpretacion"]].to_string(
                index=False
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Representatividad Index (IR) from composition residuals."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / DEFAULT_INPUT_FILE,
        help=f"Path to {DEFAULT_INPUT_FILE}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_file(args.input, DEFAULT_INPUT_FILE)

    residual_df = pd.read_csv(args.input)
    index_df = build_representativity_index(residual_df)
    summary_df = build_summary_table(index_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / OUTPUT_FILENAME
    summary_path = args.output_dir / SUMMARY_FILENAME
    index_df.to_csv(index_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    print_index_table(index_df)
    print(f"\nSaved {index_path.resolve()}")
    print(f"Saved {summary_path.resolve()}")


if __name__ == "__main__":
    main()
