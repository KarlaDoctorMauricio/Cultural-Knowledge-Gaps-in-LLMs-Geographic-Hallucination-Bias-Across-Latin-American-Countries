import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fairness_toolkit.choclo import LOCAL_SAMPLE_PATH, add_group_columns, load_choclo
from analysis.question_composition_by_country import (
    build_question_composition_by_country,
    load_questions_table,
)


def test_choclo_sample_has_fifteen_per_country():
    df = load_questions_table(LOCAL_SAMPLE_PATH)
    result = build_question_composition_by_country(df)

    assert len(result) == 18
    assert (result["n_total"] == 15).all()
    assert set(result.columns) >= {
        "pais",
        "n_total",
        "n_public_figure",
        "pct_public_figure",
        "n_dish",
        "n_object",
    }


def test_pct_public_figure_sorted_descending():
    df = load_questions_table(LOCAL_SAMPLE_PATH)
    result = build_question_composition_by_country(df)
    values = result["pct_public_figure"].tolist()
    assert values == sorted(values, reverse=True)


def test_load_from_evaluation_results_columns(tmp_path):
    sample = add_group_columns(load_choclo(LOCAL_SAMPLE_PATH)).head(30)
    path = tmp_path / "evaluation_results.csv"
    sample.to_csv(path, index=False)

    df = load_questions_table(path)
    result = build_question_composition_by_country(df)
    assert not result.empty
    assert result["n_total"].sum() == 30
