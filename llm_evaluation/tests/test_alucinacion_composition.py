import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze_alucinacion_composition import (  # noqa: E402
    DEFAULT_CLASSIFICATIONS_FILE,
    build_alucinacion_composition_analysis,
    compute_alucinacion_by_category_global,
    load_results_from_dir,
    resolve_classifications_path,
    run_alucinacion_composition_analysis,
)
from evaluator import JUDGE_PROMPT_VERSION  # noqa: E402


def _make_classified() -> pd.DataFrame:
    """Small synthetic set: hard category alucinates more globally."""
    rows = []
    categories = {
        "geography": ("correcta", "correcta", "alucinacion"),
        "public_figure": ("alucinacion", "alucinacion", "parcial"),
    }
    idx = 0
    for pais in ("chile", "peru"):
        for model in ("GPT", "Claude"):
            for category, labels in categories.items():
                for label in labels:
                    rows.append(
                        {
                            "row_index": idx,
                            "model": model,
                            "pais": pais,
                            "Category": category,
                            "judge_label": label,
                        }
                    )
                    idx += 1
    return pd.DataFrame(rows)


def _make_scored(classified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in classified.iterrows():
        rows.append(
            {
                "pais": row["pais"],
                "Category": row["Category"],
                "similarity_GPT": 0.5,
                "similarity_Claude": 0.5,
                "y_true": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _make_proportions(classified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pais, model), subset in classified.groupby(["pais", "model"]):
        n = len(subset)
        n_aluc = int((subset["judge_label"] == "alucinacion").sum())
        rows.append(
            {
                "pais": pais,
                "model": model,
                "n_total": n,
                "pct_alucinacion": round(100.0 * n_aluc / n, 2),
                "confiable": True,
            }
        )
    return pd.DataFrame(rows)


def test_compute_alucinacion_by_category_global():
    classified = _make_classified()
    global_df = compute_alucinacion_by_category_global(classified)

    assert not global_df.empty
    pub = global_df[
        (global_df["model"] == "GPT") & (global_df["category"] == "public_figure")
    ].iloc[0]
    geo = global_df[
        (global_df["model"] == "GPT") & (global_df["category"] == "geography")
    ].iloc[0]
    assert pub["pct_alucinacion"] > geo["pct_alucinacion"]


def test_build_alucinacion_composition_has_expected_columns():
    classified = _make_classified()
    proportions = _make_proportions(classified)
    scored = _make_scored(classified)
    global_df = compute_alucinacion_by_category_global(classified)

    result = build_alucinacion_composition_analysis(
        classified,
        proportions,
        scored,
        global_df,
    )

    assert {
        "pais",
        "model",
        "pct_alucinacion_observado",
        "pct_alucinacion_esperado_por_mezcla",
        "residual_alucinacion",
        "interpretacion",
    }.issubset(result.columns)
    assert not result["residual_alucinacion"].isna().all()


def test_run_alucinacion_composition_analysis_writes_csv(tmp_path):
    classified = _make_classified()
    proportions = _make_proportions(classified)
    scored = _make_scored(classified)

    outputs = run_alucinacion_composition_analysis(
        classified,
        proportions,
        scored,
        tmp_path,
    )

    assert (tmp_path / "alucinacion_composition_residual.csv").exists()
    assert (tmp_path / "alucinacion_by_category_global.csv").exists()
    assert not outputs["alucinacion_composition_residual.csv"].empty


def test_resolve_classifications_prefers_judge_final(tmp_path):
    legacy = tmp_path / "low_score_classifications.csv"
    final = tmp_path / DEFAULT_CLASSIFICATIONS_FILE
    pd.DataFrame(
        {
            "row_index": [0],
            "model": ["GPT"],
            "judge_label": ["parcial"],
            "judge_prompt_version": [JUDGE_PROMPT_VERSION],
        }
    ).to_csv(legacy, index=False)
    pd.DataFrame(
        {
            "row_index": [0],
            "model": ["GPT"],
            "judge_label": ["alucinacion"],
            "judge_prompt_version": [JUDGE_PROMPT_VERSION],
        }
    ).to_csv(final, index=False)

    path = resolve_classifications_path(tmp_path)
    assert path.name == DEFAULT_CLASSIFICATIONS_FILE
    df = pd.read_csv(path)
    assert df.iloc[0]["judge_label"] == "alucinacion"


def test_load_results_from_dir_uses_judge_final(tmp_path):
    final = tmp_path / DEFAULT_CLASSIFICATIONS_FILE
    pd.DataFrame(
        {
            "row_index": [0],
            "model": ["GPT"],
            "pais": ["chile"],
            "Category": ["geography"],
            "judge_label": ["alucinacion"],
            "judge_prompt_version": [JUDGE_PROMPT_VERSION],
        }
    ).to_csv(final, index=False)
    pd.DataFrame(
        {
            "pais": ["chile"],
            "model": ["GPT"],
            "n_total": [1],
            "pct_alucinacion": [100.0],
            "judge_source_file": [DEFAULT_CLASSIFICATIONS_FILE],
        }
    ).to_csv(tmp_path / "category_proportions_by_country.csv", index=False)
    pd.DataFrame(
        {
            "pais": ["chile"],
            "Category": ["geography"],
            "similarity_GPT": [0.1],
            "similarity_Claude": [0.2],
        }
    ).to_csv(tmp_path / "evaluation_results.csv", index=False)

    loaded = load_results_from_dir(tmp_path)
    assert loaded["classified_path"].name == DEFAULT_CLASSIFICATIONS_FILE
    assert loaded["classified"].iloc[0]["judge_label"] == "alucinacion"
