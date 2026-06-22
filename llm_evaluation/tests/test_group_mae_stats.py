import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "fairness_toolkit" / "group_mae_stats.py"
spec = importlib.util.spec_from_file_location("group_mae_stats", MODULE_PATH)
group_mae_stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(group_mae_stats)

BIAS_NOT_ROBUST_NOTE = group_mae_stats.BIAS_NOT_ROBUST_NOTE
INSUFFICIENT_SAMPLE_NOTE = group_mae_stats.INSUFFICIENT_SAMPLE_NOTE
assess_bias_robustness = group_mae_stats.assess_bias_robustness
compute_group_mae_table = group_mae_stats.compute_group_mae_table


def test_compute_group_mae_table_marks_small_groups():
    y_true = np.ones(6)
    y_pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    groups = np.array(["a", "a", "a", "a", "b", "b"])

    table = compute_group_mae_table(
        y_true,
        y_pred,
        groups,
        group_column="pais",
        method="GPT",
        config_name="test",
        group_level="pais",
        min_n=5,
        n_bootstrap=200,
        random_state=0,
    )

    row_a = table[table["pais"] == "a"].iloc[0]
    row_b = table[table["pais"] == "b"].iloc[0]

    assert row_a["n_preguntas"] == 4
    assert row_a["confiable"] == False
    assert row_a["nota"] == INSUFFICIENT_SAMPLE_NOTE
    assert row_b["n_preguntas"] == 2
    assert row_b["confiable"] == False
    assert not np.isnan(row_a["mae_margin"])


def test_unreliable_groups_sorted_after_reliable():
    y_true = np.ones(10)
    y_pred = np.linspace(0.9, 0.0, 10)
    groups = np.array(["big"] * 6 + ["small"] * 4)

    table = compute_group_mae_table(
        y_true,
        y_pred,
        groups,
        group_column="pais",
        method="GPT",
        config_name="test",
        group_level="pais",
        min_n=5,
        n_bootstrap=100,
    )

    assert table.iloc[0]["confiable"] == True
    assert table.iloc[-1]["confiable"] == False


def test_assess_bias_robustness_when_extreme_is_small_sample():
    df = pd.DataFrame(
        {
            "pais": ["chile", "mexico", "peru"],
            "mae": [0.4, 0.9, 0.5],
            "n_preguntas": [20, 2, 18],
            "confiable": [True, False, True],
            "method": ["GPT", "GPT", "GPT"],
        }
    )

    note = assess_bias_robustness(df, "GPT", min_n=5)
    assert note == BIAS_NOT_ROBUST_NOTE


def test_assess_bias_robustness_when_all_groups_reliable():
    df = pd.DataFrame(
        {
            "pais": ["chile", "mexico"],
            "mae": [0.4, 0.6],
            "n_preguntas": [10, 12],
            "confiable": [True, True],
            "method": ["GPT", "GPT"],
        }
    )

    assert assess_bias_robustness(df, "GPT", min_n=5) is None
