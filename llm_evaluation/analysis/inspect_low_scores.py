#!/usr/bin/env python
"""
Extract lowest-scoring CHOCLO examples for manual review.

For GPT and Claude, selects the 5 questions with the lowest similarity score
in each of the worst-performing countries (default: argentina, chile, panama).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "results" / "evaluation_results.csv"
DEFAULT_OUTPUT = ROOT / "data" / "results" / "low_scores_examples.csv"
DEFAULT_REVIEW_OUTPUT = ROOT / "data" / "results" / "low_scores_review.txt"

DEFAULT_COUNTRIES = ("argentina", "chile", "panama")
REVIEW_COUNTRIES = ("argentina", "chile")
REVIEW_MAX_RANK = 3
DEFAULT_MODELS = ("GPT", "Claude")
TOP_K = 5

MODEL_COLUMNS = {
    "GPT": {
        "response": "response_GPT",
        "score": "similarity_GPT",
    },
    "Claude": {
        "response": "response_Claude",
        "score": "similarity_Claude",
    },
}


def extract_low_scores(
    df: pd.DataFrame,
    countries: tuple[str, ...] = DEFAULT_COUNTRIES,
    models: tuple[str, ...] = DEFAULT_MODELS,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Return the lowest-scoring examples per country and model."""
    if "pais" not in df.columns:
        raise ValueError("Input data must include a 'pais' column.")

    rows: list[dict] = []

    for model in models:
        if model not in MODEL_COLUMNS:
            raise ValueError(f"Unsupported model: {model}")

        response_col = MODEL_COLUMNS[model]["response"]
        score_col = MODEL_COLUMNS[model]["score"]

        if response_col not in df.columns or score_col not in df.columns:
            raise ValueError(f"Missing columns for model {model}: {response_col}, {score_col}")

        for pais in countries:
            subset = df[df["pais"].astype(str).str.lower() == pais.lower()].copy()
            subset = subset[subset[score_col].notna()]
            subset = subset.sort_values(score_col, ascending=True, kind="stable").head(top_k)

            for rank, (_, row) in enumerate(subset.iterrows(), start=1):
                rows.append(
                    {
                        "pais": pais,
                        "model": model,
                        "rank_en_pais": rank,
                        "Question": row["Question"],
                        "Answer": row["Answer"],
                        "response": row[response_col],
                        "score": row[score_col],
                        "Category": row["Category"],
                        "Difficulty": row["Difficulty"],
                    }
                )

    return pd.DataFrame(rows)


def _clean_text(value) -> str:
    if pd.isna(value):
        return "(sin respuesta)"
    return str(value).strip()


def format_manual_review_block(row: pd.Series) -> str:
    """Format one example as readable text for manual inspection."""
    response_label = f"response_{row['model']}"
    header = (
        f"{'=' * 80}\n"
        f"{row['pais'].upper()} | {row['model']} | rank {row['rank_en_pais']} | "
        f"score {row['score']:.4f} | {row['Category']} / {row['Difficulty']}\n"
        f"{'=' * 80}"
    )
    body = (
        f"\nPREGUNTA:\n{_clean_text(row['Question'])}\n\n"
        f"RESPUESTA ESPERADA (Answer):\n{_clean_text(row['Answer'])}\n\n"
        f"{response_label.upper()}:\n{_clean_text(row['response'])}\n"
    )
    return f"{header}{body}"


def build_manual_review(
    examples: pd.DataFrame,
    countries: tuple[str, ...] = REVIEW_COUNTRIES,
    max_rank: int = REVIEW_MAX_RANK,
) -> str:
    """Build readable review text for low-rank examples in selected countries."""
    subset = examples[
        examples["pais"].isin(countries) & (examples["rank_en_pais"] <= max_rank)
    ].copy()
    subset = subset.sort_values(
        ["pais", "model", "rank_en_pais"],
        ascending=[True, True, True],
        kind="stable",
    )

    if subset.empty:
        return "No hay ejemplos para revisión manual con los filtros indicados.\n"

    blocks = [
        "REVISION MANUAL — peores scores (rank 1-3) en argentina y chile\n",
        f"Total de casos: {len(subset)}\n",
    ]
    blocks.extend(format_manual_review_block(row) for _, row in subset.iterrows())
    return "\n".join(blocks) + "\n"


def export_manual_review(
    examples: pd.DataFrame,
    output_path: Path,
    countries: tuple[str, ...] = REVIEW_COUNTRIES,
    max_rank: int = REVIEW_MAX_RANK,
) -> pd.DataFrame:
    """Write readable review file and return the filtered subset."""
    subset = examples[
        examples["pais"].isin(countries) & (examples["rank_en_pais"] <= max_rank)
    ].copy()
    subset = subset.sort_values(
        ["pais", "model", "rank_en_pais"],
        ascending=[True, True, True],
        kind="stable",
    )

    review_text = build_manual_review(
        examples,
        countries=countries,
        max_rank=max_rank,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(review_text, encoding="utf-8")
    return subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export low-score CHOCLO examples for manual inspection."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Evaluation results CSV (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=DEFAULT_REVIEW_OUTPUT,
        help=(
            "Readable text export for argentina/chile ranks 1-3 "
            f"(default: {DEFAULT_REVIEW_OUTPUT})."
        ),
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=list(DEFAULT_COUNTRIES),
        help="Countries to inspect (default: argentina chile panama).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Number of lowest-score questions per country/model (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows.")

    examples = extract_low_scores(
        df,
        countries=tuple(args.countries),
        top_k=args.top_k,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(args.output, index=False, encoding="utf-8")

    review_subset = export_manual_review(examples, args.review_output)
    review_text = build_manual_review(examples)

    print(f"Saved {len(examples):,} examples to {args.output}")
    print(
        f"Saved manual review ({len(review_subset):,} cases, "
        f"argentina/chile rank 1-{REVIEW_MAX_RANK}) to {args.review_output}"
    )
    print("\n" + review_text)
    print("\nCSV preview:")
    preview_cols = ["pais", "model", "rank_en_pais", "score", "Category", "Difficulty"]
    print(examples[preview_cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
