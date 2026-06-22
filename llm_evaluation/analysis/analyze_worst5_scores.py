#!/usr/bin/env python
"""
Worst-5 similarity examples per country x model (read-only post-hoc analysis).

Reads existing evaluation CSVs; does not modify the evaluation pipeline.

Outputs:
  - worst5_category_distribution.csv  category mix of bottom-5 scores (all countries)
  - worst5_examples_full.csv          full text for top bias-alucinacion countries only
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

DEFAULT_MODELS = ("GPT", "Claude")
TOP_K = 5
DETAIL_COUNTRY_COUNT = 3

MODEL_COLUMNS = {
    "GPT": {"response": "response_GPT", "similarity": "similarity_GPT"},
    "Claude": {"response": "response_Claude", "similarity": "similarity_Claude"},
}

CATEGORY_COLUMNS = (
    "dish",
    "tradition",
    "public_figure",
    "geography",
    "flora",
    "fauna",
    "object",
)

DISTRIBUTION_FILENAME = "worst5_category_distribution.csv"
EXAMPLES_FILENAME = "worst5_examples_full.csv"


def load_evaluation_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "pais" not in df.columns:
        raise ValueError(f"{path} must include a pais column.")
    return df.reset_index().rename(columns={"index": "row_index"})


def load_classifications(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["row_index", "model", "judge_label"])
    return pd.read_csv(path)[["row_index", "model", "judge_label"]]


def select_high_bias_countries(
    fairness_gaps: pd.DataFrame,
    proportions: pd.DataFrame,
    n: int = DETAIL_COUNTRY_COUNT,
) -> list[str]:
    """
    Pick countries for the detailed examples file.

    Starts with ``max_pais_alucinacion`` from fairness gaps, then fills with
    the next highest country-level hallucination rates.
    """
    priority: list[str] = []
    for pais in fairness_gaps["max_pais_alucinacion"].dropna():
        pais_str = str(pais)
        if pais_str not in priority:
            priority.append(pais_str)

    max_by_pais = (
        proportions.groupby("pais")["pct_alucinacion"].max().sort_values(ascending=False)
    )
    for pais in max_by_pais.index:
        if str(pais) not in priority:
            priority.append(str(pais))
        if len(priority) >= n:
            break

    return priority[:n]


def extract_worst5_per_country_model(
    df: pd.DataFrame,
    models: tuple[str, ...] = DEFAULT_MODELS,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Bottom ``top_k`` similarity scores for every country x model."""
    countries = sorted(df["pais"].astype(str).unique())
    rows: list[dict] = []

    for model in models:
        cols = MODEL_COLUMNS[model]
        response_col = cols["response"]
        similarity_col = cols["similarity"]

        if response_col not in df.columns or similarity_col not in df.columns:
            raise ValueError(f"Missing columns for {model}: {response_col}, {similarity_col}")

        for pais in countries:
            subset = df[df["pais"].astype(str) == pais].copy()
            subset = subset[subset[similarity_col].notna()]
            subset = subset.sort_values(similarity_col, ascending=True, kind="stable").head(top_k)

            for rank, (_, row) in enumerate(subset.iterrows(), start=1):
                rows.append(
                    {
                        "pais": pais,
                        "model": model,
                        "rank_en_pais": rank,
                        "row_index": int(row["row_index"]),
                        "Category": row["Category"],
                        "Difficulty": row.get("Difficulty"),
                        "Question": row["Question"],
                        "Answer": row["Answer"],
                        "response": row[response_col],
                        "similarity": float(row[similarity_col]),
                    }
                )

    return pd.DataFrame(rows)


def attach_judge_labels(worst5: pd.DataFrame, classifications: pd.DataFrame) -> pd.DataFrame:
    if classifications.empty:
        worst5 = worst5.copy()
        worst5["judge_label"] = pd.NA
        return worst5

    merged = worst5.merge(classifications, on=["row_index", "model"], how="left")
    return merged


def _category_breakdown_text(counts: dict[str, int]) -> str:
    parts = [f"{n} {cat}" for cat, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if n]
    return ", ".join(parts)


def build_worst5_category_distribution(worst5: pd.DataFrame) -> pd.DataFrame:
    """Aggregate category counts for each country x model worst-5 set."""
    rows: list[dict] = []

    for (pais, model), subset in worst5.groupby(["pais", "model"], sort=True):
        counts = subset["Category"].astype(str).value_counts().to_dict()
        n_total = int(len(subset))

        row = {
            "pais": pais,
            "model": model,
            "n_worst": n_total,
            **{f"n_{cat}": int(counts.get(cat, 0)) for cat in CATEGORY_COLUMNS},
            "category_breakdown": _category_breakdown_text(
                {cat: int(counts.get(cat, 0)) for cat in CATEGORY_COLUMNS}
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["model", "pais"], kind="stable").reset_index(drop=True)


def build_worst5_examples_full(
    worst5: pd.DataFrame,
    detail_countries: list[str],
) -> pd.DataFrame:
    """Full question/answer/response rows for selected high-bias countries."""
    subset = worst5[worst5["pais"].isin(detail_countries)].copy()
    subset = subset.sort_values(
        ["pais", "model", "rank_en_pais"],
        ascending=[True, True, True],
        kind="stable",
    )

    columns = [
        "pais",
        "model",
        "rank_en_pais",
        "Category",
        "Difficulty",
        "similarity",
        "judge_label",
        "Question",
        "Answer",
        "response",
    ]
    for col in columns:
        if col not in subset.columns:
            subset[col] = pd.NA

    return subset[columns].reset_index(drop=True)


def run_worst5_analysis(
    evaluation_path: Path,
    output_dir: Path,
    *,
    classifications_path: Path | None = None,
    fairness_gaps_path: Path | None = None,
    proportions_path: Path | None = None,
    models: tuple[str, ...] = DEFAULT_MODELS,
    top_k: int = TOP_K,
    detail_country_count: int = DETAIL_COUNTRY_COUNT,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    evaluation_path = Path(evaluation_path)

    classifications_path = classifications_path or output_dir / "low_score_classifications.csv"
    fairness_gaps_path = fairness_gaps_path or output_dir / "category_proportions_fairness_gaps.csv"
    proportions_path = proportions_path or output_dir / "category_proportions_by_country.csv"

    df = load_evaluation_results(evaluation_path)
    classifications = load_classifications(classifications_path)
    worst5 = extract_worst5_per_country_model(df, models=models, top_k=top_k)
    worst5 = attach_judge_labels(worst5, classifications)

    distribution = build_worst5_category_distribution(worst5)

    if fairness_gaps_path.exists() and proportions_path.exists():
        fairness_gaps = pd.read_csv(fairness_gaps_path)
        proportions = pd.read_csv(proportions_path)
        detail_countries = select_high_bias_countries(
            fairness_gaps,
            proportions,
            n=detail_country_count,
        )
    else:
        detail_countries = sorted(df["pais"].astype(str).unique())[:detail_country_count]

    examples = build_worst5_examples_full(worst5, detail_countries)

    output_dir.mkdir(parents=True, exist_ok=True)
    distribution.to_csv(output_dir / DISTRIBUTION_FILENAME, index=False, encoding="utf-8")
    examples.to_csv(output_dir / EXAMPLES_FILENAME, index=False, encoding="utf-8")

    return {
        DISTRIBUTION_FILENAME: distribution,
        EXAMPLES_FILENAME: examples,
        "_detail_countries": detail_countries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze worst-5 similarity scores per country and model."
    )
    parser.add_argument(
        "--evaluation-results",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "evaluation_results.csv",
        help="Path to evaluation_results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSVs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Number of lowest-similarity questions per country-model (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_worst5_analysis(
        args.evaluation_results,
        args.output_dir,
        top_k=args.top_k,
    )

    detail = outputs["_detail_countries"]
    distribution = outputs[DISTRIBUTION_FILENAME]
    examples = outputs[EXAMPLES_FILENAME]

    print(f"Saved {len(distribution):,} rows to {args.output_dir / DISTRIBUTION_FILENAME}")
    print(f"Saved {len(examples):,} rows to {args.output_dir / EXAMPLES_FILENAME}")
    print(f"Detail countries (high bias_alucinacion): {', '.join(detail)}")

    print("\nSample distribution (argentina):")
    sample = distribution[distribution["pais"] == "argentina"][["model", "category_breakdown"]]
    if not sample.empty:
        print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
