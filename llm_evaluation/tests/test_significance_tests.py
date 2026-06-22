import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.significance_tests import (
    benjamini_hochberg,
    ci_overlap,
    run_global_alucinacion_tests,
    run_multiple_comparisons,
    run_paired_error_tests,
)


def test_benjamini_hochberg_monotonic():
    corrected = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert corrected[0] <= corrected[1] <= corrected[2] <= corrected[3]
    assert corrected[0] <= 0.05


def test_ci_overlap():
    assert not ci_overlap(10, 20, 25, 30)
    assert ci_overlap(10, 25, 20, 30)


def test_paired_error_tests_on_synthetic():
    n = 30
    df = pd.DataFrame(
        {
            "error_GPT": np.linspace(0.9, 0.6, n),
            "error_Claude": np.linspace(0.4, 0.1, n),
        }
    )
    result = run_paired_error_tests(df)
    assert "wilcoxon_signed_rank_error" in result["test_name"].values
    wilcoxon = result[result["test_name"] == "wilcoxon_signed_rank_error"].iloc[0]
    assert wilcoxon["p_value"] < 0.05
    assert wilcoxon["effect_size"] > 0


def test_mcnemar_on_synthetic():
    rows = []
    for i in range(20):
        rows.append(
            {
                "row_index": i,
                "model": "GPT",
                "judge_label": "alucinacion" if i < 10 else "correcta",
            }
        )
        rows.append(
            {
                "row_index": i,
                "model": "Claude",
                "judge_label": "alucinacion" if i < 3 else "correcta",
            }
        )
    classifications = pd.DataFrame(rows)
    result = run_global_alucinacion_tests(classifications)
    mcnemar = result[result["test_name"] == "mcnemar_alucinacion_paired"].iloc[0]
    assert mcnemar["p_value"] < 0.05


def test_multiple_comparisons_schema():
    proportions = pd.DataFrame(
        {
            "pais": ["a", "b", "c"],
            "model": ["GPT", "GPT", "GPT"],
            "n_alucinacion": [7, 1, 3],
            "n_total": [15, 15, 15],
            "pct_alucinacion": [46.67, 6.67, 20.0],
            "pct_alucinacion_ci_lower": [20.0, 0.0, 0.0],
            "pct_alucinacion_ci_upper": [73.0, 20.0, 40.0],
        }
    )
    gaps = pd.DataFrame(
        {
            "model": ["GPT"],
            "max_pais_alucinacion": ["a"],
            "min_pais_alucinacion": ["b"],
        }
    )
    result = run_multiple_comparisons(proportions, gaps)
    assert "comparacion" in result.columns
    assert "p_value_corrected" in result.columns
    assert "significativo_post_correccion" in result.columns
    highlight = result[result["fairness_gap_pair"]]
    assert not highlight.empty
