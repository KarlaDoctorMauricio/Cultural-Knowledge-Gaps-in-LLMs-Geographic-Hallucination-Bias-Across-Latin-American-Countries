import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze_category_composition import (  # noqa: E402
    build_country_composition_analysis,
    compute_category_mae_by_country,
    run_category_composition_analysis,
    rollup_interseccion_by_category,
    summarize_category_difficulty,
)
from fairness_toolkit.choclo import LOCAL_SAMPLE_PATH, load_choclo, sample_choclo
from fairness_toolkit.group_mae_stats import build_group_mae_category_tables, build_group_mae_tables
from fairness_toolkit.llm_clients import MockLLMClient
from fairness_toolkit.pipeline import generate_responses, score_responses


class FixedSimilarityScorer:
    def score(self, predictions, references):
        scores = []
        for pred, ref in zip(predictions, references):
            if pred == ref:
                scores.append(1.0)
            elif pred and ref and pred in ref:
                scores.append(0.8)
            else:
                scores.append(0.4)
        return np.array(scores, dtype=float)


@pytest.fixture
def scored_sample():
    df = sample_choclo(load_choclo(LOCAL_SAMPLE_PATH), n=30, random_state=3)
    clients = {name: MockLLMClient(name) for name in ["GPT", "Claude"]}
    responses = generate_responses(df, clients=clients, model_names=["GPT", "Claude"])
    scored, preds = score_responses(
        responses,
        model_names=["GPT", "Claude"],
        scorer=FixedSimilarityScorer(),
    )
    return scored, preds


def test_build_group_mae_category_tables(scored_sample):
    scored, preds = scored_sample
    table = build_group_mae_category_tables(
        scored["y_true"].to_numpy(),
        preds,
        scored["Category"].astype(str).to_numpy(),
        "CHOCLO_test",
        min_n=1,
        n_bootstrap=50,
    )

    assert not table.empty
    assert set(table["category"].unique()).issubset(set(scored["Category"].unique()))
    assert set(table["method"]) == {"GPT", "Claude"}
    assert "group_level" in table.columns
    assert (table["group_level"] == "category").all()


def test_category_composition_analysis_outputs(scored_sample):
    scored, preds = scored_sample
    y_true = scored["y_true"].to_numpy()
    pais = scored["pais"].to_numpy()
    interseccion = scored["interseccion"].to_numpy()
    category = scored["Category"].astype(str).to_numpy()

    df_pais, df_inter = build_group_mae_tables(
        y_true, preds, pais, interseccion, "CHOCLO_test", min_n=1, n_bootstrap=50
    )
    df_category = build_group_mae_category_tables(
        y_true, preds, category, "CHOCLO_test", min_n=1, n_bootstrap=50
    )

    outputs = run_category_composition_analysis(
        scored,
        df_pais,
        df_category,
        df_inter,
    )

    assert not outputs["category_composition_analysis.csv"].empty
    assert not outputs["category_mae_by_country.csv"].empty
    assert not outputs["category_difficulty_summary.csv"].empty
    assert not outputs["interseccion_by_category.csv"].empty


def test_build_country_composition_analysis_has_expected_columns(scored_sample):
    scored, preds = scored_sample
    df_category = build_group_mae_category_tables(
        scored["y_true"].to_numpy(),
        preds,
        scored["Category"].astype(str).to_numpy(),
        "CHOCLO_test",
        min_n=1,
        n_bootstrap=50,
    )
    df_pais, _ = build_group_mae_tables(
        scored["y_true"].to_numpy(),
        preds,
        scored["pais"].to_numpy(),
        scored["interseccion"].to_numpy(),
        "CHOCLO_test",
        min_n=1,
        n_bootstrap=50,
    )

    result = build_country_composition_analysis(scored, df_pais, df_category)
    assert {
        "pais",
        "method",
        "mae_observed",
        "mae_expected_from_category_mix",
        "mae_composition_residual",
        "hard_category_overrepresentation",
        "interpretacion",
    }.issubset(result.columns)


def test_rollup_interseccion_by_category(scored_sample):
    scored, preds = scored_sample
    _, df_inter = build_group_mae_tables(
        scored["y_true"].to_numpy(),
        preds,
        scored["pais"].to_numpy(),
        scored["interseccion"].to_numpy(),
        "CHOCLO_test",
        min_n=1,
        n_bootstrap=50,
    )
    rolled = rollup_interseccion_by_category(df_inter, scored)
    assert not rolled.empty
    assert "category" in rolled.columns
    assert "mae_mean" in rolled.columns
