import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze_worst5_scores import (
    build_worst5_category_distribution,
    build_worst5_examples_full,
    extract_worst5_per_country_model,
    select_high_bias_countries,
)


def _mini_eval_df() -> pd.DataFrame:
    rows = []
    idx = 0
    for pais in ("argentina", "chile", "peru"):
        for i, (cat, sim_gpt, sim_claude) in enumerate(
            [
                ("geography", 0.1, 0.2),
                ("tradition", 0.15, 0.25),
                ("fauna", 0.3, 0.35),
                ("dish", 0.5, 0.55),
                ("flora", 0.6, 0.65),
            ]
        ):
            rows.append(
                {
                    "row_index": idx,
                    "pais": pais,
                    "Category": cat,
                    "Question": f"Q {pais} {i}",
                    "Answer": "A",
                    "response_GPT": "RG",
                    "response_Claude": "RC",
                    "similarity_GPT": sim_gpt,
                    "similarity_Claude": sim_claude,
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def test_extract_worst5_returns_five_per_country_model():
    df = _mini_eval_df()
    worst5 = extract_worst5_per_country_model(df, top_k=5)

    assert len(worst5) == 3 * 2 * 5
    assert (worst5.groupby(["pais", "model"]).size() == 5).all()
    arg_gpt = worst5[(worst5["pais"] == "argentina") & (worst5["model"] == "GPT")]
    assert set(arg_gpt["Category"]) == {"geography", "tradition", "fauna", "dish", "flora"}
    assert arg_gpt.nsmallest(1, "similarity")["Category"].iloc[0] == "geography"


def test_distribution_has_category_counts():
    df = _mini_eval_df()
    worst5 = extract_worst5_per_country_model(df, top_k=5)
    distribution = build_worst5_category_distribution(worst5)

    assert {"pais", "model", "n_worst", "n_geography", "category_breakdown"}.issubset(
        distribution.columns
    )
    row = distribution[
        (distribution["pais"] == "argentina") & (distribution["model"] == "GPT")
    ].iloc[0]
    assert row["n_worst"] == 5
    assert row["n_geography"] == 1


def test_select_high_bias_countries():
    gaps = pd.DataFrame(
        {
            "model": ["GPT", "Claude"],
            "max_pais_alucinacion": ["argentina", "venezuela"],
        }
    )
    props = pd.DataFrame(
        {
            "pais": ["argentina", "venezuela", "guatemala", "mexico"] * 2,
            "model": ["GPT"] * 4 + ["Claude"] * 4,
            "pct_alucinacion": [46.67, 40.0, 46.67, 46.67, 13.33, 40.0, 33.33, 20.0],
        }
    )
    countries = select_high_bias_countries(gaps, props, n=3)
    assert set(countries) == {"argentina", "venezuela", "guatemala"}


def test_examples_full_filters_countries():
    df = _mini_eval_df()
    worst5 = extract_worst5_per_country_model(df, top_k=5)
    examples = build_worst5_examples_full(worst5, ["argentina", "chile"])

    assert set(examples["pais"].unique()) == {"argentina", "chile"}
    assert len(examples) == 2 * 2 * 5
    assert "judge_label" in examples.columns or "Question" in examples.columns
