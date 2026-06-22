import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fairness_toolkit.choclo import LOCAL_SAMPLE_PATH, add_group_columns, load_choclo, sample_choclo
from fairness_toolkit.llm_clients import MockLLMClient, get_model_clients
from fairness_toolkit.pipeline import (
    build_group_mae_tables,
    evaluate_available_models,
    generate_responses,
    run_choclo_evaluation,
    score_responses,
)


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
def choclo_sample():
    return sample_choclo(load_choclo(LOCAL_SAMPLE_PATH), n=12, random_state=7)


def test_add_group_columns():
    df = pd.DataFrame(
        {
            "Country": ["Chile", "Mexico"],
            "Category": ["Musica", "Arte"],
            "Difficulty": ["easy", "hard"],
        }
    )
    grouped = add_group_columns(df)
    assert grouped["pais"].tolist() == ["chile", "mexico"]
    assert grouped["interseccion"].tolist() == [
        "chile_Musica_easy",
        "mexico_Arte_hard",
    ]


def test_generate_responses_adds_three_models(choclo_sample):
    clients = {name: MockLLMClient(name) for name in ["LatamGPT", "GPT", "Claude"]}
    result = generate_responses(choclo_sample, clients=clients)
    for model in clients:
        assert f"response_{model}" in result.columns


def test_score_responses_produces_similarity_columns(choclo_sample):
    clients = {name: MockLLMClient(name) for name in ["LatamGPT", "GPT", "Claude"]}
    responses = generate_responses(choclo_sample, clients=clients)
    scored, preds = score_responses(responses, scorer=FixedSimilarityScorer())

    assert len(preds) == 3
    assert "similarity_LatamGPT" in scored.columns
    assert np.allclose(scored["y_true"].unique(), [1.0])


def test_build_group_mae_tables_schema(choclo_sample):
    clients = {name: MockLLMClient(name) for name in ["LatamGPT", "GPT", "Claude"]}
    responses = generate_responses(choclo_sample, clients=clients)
    scored, preds = score_responses(responses, scorer=FixedSimilarityScorer())

    y_true = scored["y_true"].to_numpy()
    pais = scored["pais"].to_numpy()
    interseccion = scored["interseccion"].to_numpy()

    df_pais, df_inter = build_group_mae_tables(
        y_true, preds, pais, interseccion, "CHOCLO_test"
    )

    assert set(df_pais.columns) == {
        "pais",
        "mae",
        "n_preguntas",
        "confiable",
        "nota",
        "mae_ci_lower",
        "mae_ci_upper",
        "mae_margin",
        "method",
        "config",
        "group_level",
    }
    assert set(df_inter.columns) == {
        "interseccion",
        "mae",
        "n_preguntas",
        "confiable",
        "nota",
        "mae_ci_lower",
        "mae_ci_upper",
        "mae_margin",
        "method",
        "config",
        "group_level",
    }


def test_score_responses_handles_unavailable_latamgpt(choclo_sample):
    df = add_group_columns(choclo_sample.copy())
    df["response_LatamGPT"] = None
    df["note_LatamGPT"] = (
        "LatamGPT no disponible: modelo sin Inference Provider activo en Hugging Face"
    )
    df["response_GPT"] = df["Answer"]
    df["note_GPT"] = None
    df["response_Claude"] = df["Answer"]
    df["note_Claude"] = None

    scored, preds = score_responses(df, scorer=FixedSimilarityScorer())

    assert "LatamGPT" not in preds
    assert "GPT" in preds
    assert "Claude" in preds
    assert scored["similarity_LatamGPT"].isna().all()
    assert scored["note_LatamGPT"].notna().all()


def test_evaluate_available_models_skips_unavailable():
    y_true = np.ones(3)
    pais = np.array(["chile", "chile", "mexico"])
    preds = {
        "LatamGPT": np.array([np.nan, np.nan, np.nan]),
        "GPT": np.array([0.9, 0.8, 0.7]),
    }

    metrics = evaluate_available_models(preds, y_true, pais, "CHOCLO_test")
    by_method = {row["method"]: row for row in metrics}

    assert np.isnan(by_method["LatamGPT"]["mae"])
    assert by_method["LatamGPT"]["note"] == "Modelo no disponible"
    assert by_method["GPT"]["n_valid"] == 3


def test_run_choclo_evaluation_end_to_end(choclo_sample):
    clients = get_model_clients(use_mock=True)
    results = run_choclo_evaluation(
        choclo_sample,
        config_name="CHOCLO_test",
        clients=clients,
        scorer=FixedSimilarityScorer(),
        apply_calibration=True,
    )

    assert not results["df_metrics"].empty
    assert not results["df_group_mae_pais"].empty
    assert not results["df_group_mae_interseccion"].empty
    assert not results["df_group_mae_category"].empty
    assert not results["df_metrics_post"].empty
    assert results["best_method"] in {"LatamGPT", "GPT", "Claude"}
    assert any(name.endswith("_post") for name in results["preds_post"])
