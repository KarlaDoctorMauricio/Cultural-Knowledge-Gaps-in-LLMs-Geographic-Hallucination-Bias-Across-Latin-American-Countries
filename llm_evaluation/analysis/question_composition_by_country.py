#!/usr/bin/env python
"""
Build an exact category composition table for the evaluated CHOCLO sample.

Inspection only — does not modify the evaluation or fairness pipeline.

Example:
    python question_composition_by_country.py
    python question_composition_by_country.py --input data/results/evaluation_results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairness_toolkit.choclo import LOCAL_SAMPLE_PATH, add_group_columns, load_choclo  # noqa: E402
from results_io import DEFAULT_OUTPUT_DIR  # noqa: E402

OUTPUT_FILENAME = "question_composition_by_country.csv"

CATEGORY_COLUMNS = (
    "dish",
    "tradition",
    "public_figure",
    "geography",
    "flora",
    "fauna",
    "object",
)


def load_questions_table(path: Path) -> pd.DataFrame:
    """Load sample or evaluation results and ensure ``pais`` + ``Category``."""
    df = load_choclo(path) if path.suffix.lower() == ".csv" else pd.read_csv(path)

    if "pais" not in df.columns:
        if "Country" not in df.columns:
            raise ValueError(f"{path} must include Country or pais.")
        df = add_group_columns(df)
    elif "Category" not in df.columns:
        raise ValueError(f"{path} must include Category.")

    if "Category" not in df.columns:
        raise ValueError(f"{path} must include Category.")

    return df[["pais", "Category"]].copy()


def build_question_composition_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """One row per country with exact category counts and public_figure share."""
    working = df.copy()
    working["Category"] = working["Category"].astype(str)

    unknown = sorted(set(working["Category"]) - set(CATEGORY_COLUMNS))
    if unknown:
        raise ValueError(f"Unexpected categories in input: {unknown}")

    counts = (
        working.groupby(["pais", "Category"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CATEGORY_COLUMNS, fill_value=0)
    )

    rows: list[dict] = []
    for pais, row in counts.iterrows():
        n_by_cat = {cat: int(row[cat]) for cat in CATEGORY_COLUMNS}
        n_total = int(sum(n_by_cat.values()))
        pct_public_figure = round(100.0 * n_by_cat["public_figure"] / n_total, 2) if n_total else 0.0

        entry: dict = {
            "pais": pais,
            "n_total": n_total,
            **{f"n_{cat}": n_by_cat[cat] for cat in CATEGORY_COLUMNS},
            "pct_public_figure": pct_public_figure,
        }
        rows.append(entry)

    result = pd.DataFrame(rows)
    return result.sort_values("pct_public_figure", ascending=False, kind="stable").reset_index(
        drop=True
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export exact CHOCLO category composition by country for the evaluated sample."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=LOCAL_SAMPLE_PATH,
        help=f"Input CSV with Country/pais and Category (default: {LOCAL_SAMPLE_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for {OUTPUT_FILENAME} (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_questions_table(args.input)
    result = build_question_composition_by_country(df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / OUTPUT_FILENAME
    result.to_csv(output_path, index=False, encoding="utf-8")

    global_pct = round(
        100.0 * result["n_public_figure"].sum() / result["n_total"].sum(),
        2,
    )
    print(f"Loaded {len(df):,} questions from {args.input}")
    print(f"Promedio global pct_public_figure: {global_pct}%")
    print(f"Saved {len(result):,} countries to {output_path.resolve()}")
    print("\nTop 5 by pct_public_figure:")
    print(
        result[["pais", "n_total", "n_public_figure", "pct_public_figure"]]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
