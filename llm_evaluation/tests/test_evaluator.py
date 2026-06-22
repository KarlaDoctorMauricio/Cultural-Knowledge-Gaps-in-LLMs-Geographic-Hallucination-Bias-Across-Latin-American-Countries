import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator import (  # noqa: E402
    QUALITY_LABELS,
    build_response_quality_breakdown_table,
    build_response_quality_by_country_table,
    classify_low_score_responses,
    ensure_embedding_score_columns,
    llm_judge_classify,
    rule_based_judge_classify,
)


def test_rule_based_detects_abstention():
    response = (
        "No tengo informacion especifica sobre Oscar Alem. "
        "Podrias proporcionar mas detalles?"
    )
    assert rule_based_judge_classify(response) == "abstencion"


def test_rule_based_uses_score_for_correcta_and_parcial():
    assert rule_based_judge_classify("Una respuesta concreta.", score=0.9) == "correcta"
    assert rule_based_judge_classify("Una respuesta concreta.", score=0.55) == "parcial"
    assert rule_based_judge_classify("Un dato incorrecto.", score=0.1) == "alucinacion"


def test_normalize_all_four_labels_via_llm_judge():
    cases = {
        "CORRECTA": "correcta",
        "PARCIAL": "parcial",
        "ALUCINACION": "alucinacion",
        "ABSTENCION": "abstencion",
    }
    for raw, expected in cases.items():
        label, _, method = llm_judge_classify(
            question="Q",
            reference="A",
            response="R",
            score=0.5,
            primary_query_fn=lambda _prompt, raw=raw, **_kw: raw,
            backup_query_fn=lambda _prompt, **_kw: pytest.fail("Claude should not be called"),
        )
        assert label == expected
        assert method == "gpt"


def test_ensure_embedding_score_columns_aliases_similarity():
    df = pd.DataFrame({"similarity_GPT": [0.1, 0.9], "similarity_Claude": [0.2, 0.8]})
    out = ensure_embedding_score_columns(df)
    assert out["score_embedding_GPT"].tolist() == [0.1, 0.9]
    assert out["score_embedding_Claude"].tolist() == [0.2, 0.8]


def test_classify_low_score_responses_includes_all_valid_rows():
    df = pd.DataFrame(
        {
            "Question": ["Q1", "Q2", "Q3"],
            "Answer": ["A1", "A2", "A3"],
            "response_GPT": [
                "No se la respuesta exacta.",
                "Un dato concreto incorrecto.",
                "Respuesta correcta.",
            ],
            "similarity_GPT": [0.1, 0.25, 0.95],
            "response_Claude": [
                "No se la respuesta exacta.",
                "Un dato concreto incorrecto.",
                "Respuesta correcta.",
            ],
            "similarity_Claude": [0.15, 0.29, 0.9],
        }
    )

    classified = classify_low_score_responses(
        df,
        model_names=("GPT", "Claude"),
        use_llm_judge=False,
    )

    assert len(classified) == 6
    assert set(classified["model"]) == {"GPT", "Claude"}
    assert (classified["judge_method"] == "heuristic_fallback").all()
    assert set(classified["judge_label"]).issubset(set(QUALITY_LABELS))


def test_llm_judge_classify_falls_back_to_claude():
    label, raw, method = llm_judge_classify(
        question="Q",
        reference="A",
        response="No tengo informacion.",
        score=0.2,
        primary_query_fn=lambda _prompt, **_kw: None,
        backup_query_fn=lambda _prompt, **_kw: "PARCIAL",
    )
    assert label == "parcial"
    assert raw == "PARCIAL"
    assert method == "claude"


def test_llm_judge_classify_uses_heuristic_when_both_judges_fail():
    label, raw, method = llm_judge_classify(
        question="Q",
        reference="A",
        response="No tengo informacion suficiente.",
        score=0.2,
        primary_query_fn=lambda _prompt, **_kw: None,
        backup_query_fn=lambda _prompt, **_kw: "respuesta ambigua",
    )
    assert label == "abstencion"
    assert raw == "respuesta ambigua"
    assert method == "heuristic_fallback"


def test_build_response_quality_breakdown_table_percentages():
    classified = pd.DataFrame(
        {
            "model": ["GPT", "GPT", "GPT", "GPT", "Claude", "Claude"],
            "judge_label": [
                "correcta",
                "parcial",
                "alucinacion",
                "abstencion",
                "parcial",
                "abstencion",
            ],
        }
    )
    summary = build_response_quality_breakdown_table(classified)

    gpt = summary[summary["model"] == "GPT"].iloc[0]
    claude = summary[summary["model"] == "Claude"].iloc[0]

    assert gpt["n_judged"] == 4
    assert gpt["n_correcta"] == 1
    assert gpt["n_parcial"] == 1
    assert gpt["n_alucinacion"] == 1
    assert gpt["n_abstencion"] == 1
    assert gpt["pct_correcta"] == 25.0
    assert gpt["pct_parcial"] == 25.0

    assert claude["n_judged"] == 2
    assert claude["pct_abstencion"] == 50.0


def test_build_response_quality_by_country_table():
    classified = pd.DataFrame(
        {
            "pais": ["argentina", "argentina", "chile", "chile", "chile"],
            "model": ["GPT", "GPT", "GPT", "Claude", "Claude"],
            "judge_label": [
                "alucinacion",
                "abstencion",
                "correcta",
                "parcial",
                "abstencion",
            ],
        }
    )
    by_country = build_response_quality_by_country_table(classified)

    gpt_ar = by_country[
        (by_country["model"] == "GPT") & (by_country["pais"] == "argentina")
    ].iloc[0]
    claude_cl = by_country[
        (by_country["model"] == "Claude") & (by_country["pais"] == "chile")
    ].iloc[0]

    assert gpt_ar["n_judged"] == 2
    assert gpt_ar["pct_alucinacion"] == 50.0
    assert claude_cl["pct_parcial"] == 50.0
    assert claude_cl["pct_abstencion"] == 50.0
    assert len(by_country) == 3
