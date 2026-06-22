import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.category_proportions_by_country import (  # noqa: E402
    compute_category_proportions_by_country,
    compute_fairness_gaps,
    plot_category_proportions_chart,
)
from fairness_toolkit.group_mae_stats import _bootstrap_proportion  # noqa: E402


def _make_classified() -> pd.DataFrame:
    rows = []
    for pais, aluc, abst, corr, parc in [
        ("chile", 8, 1, 1, 0),
        ("mexico", 2, 1, 6, 1),
        ("peru", 5, 2, 2, 1),
    ]:
        for model in ("GPT", "Claude"):
            for _ in range(aluc):
                rows.append({"pais": pais, "model": model, "judge_label": "alucinacion"})
            for _ in range(abst):
                rows.append({"pais": pais, "model": model, "judge_label": "abstencion"})
            for _ in range(corr):
                rows.append({"pais": pais, "model": model, "judge_label": "correcta"})
            for _ in range(parc):
                rows.append({"pais": pais, "model": model, "judge_label": "parcial"})
    return pd.DataFrame(rows)


def test_proportions_sum_to_100():
    table = compute_category_proportions_by_country(
        _make_classified(), min_n=1, n_bootstrap=50, random_state=7
    )
    for _, row in table.iterrows():
        total = sum(row[f"pct_{label}"] for label in ("correcta", "parcial", "alucinacion", "abstencion"))
        assert abs(total - 100.0) < 0.01


def test_all_label_bootstrap_ci_columns_present():
    table = compute_category_proportions_by_country(
        _make_classified(), min_n=1, n_bootstrap=50, random_state=7
    )
    for label in ("correcta", "parcial", "alucinacion", "abstencion"):
        lower = table[f"pct_{label}_ci_lower"]
        upper = table[f"pct_{label}_ci_upper"]
        assert lower.notna().all()
        assert upper.notna().all()
        assert (lower <= upper).all()
        assert (lower >= 0).all() and (upper <= 100).all()


def test_fairness_gaps_primary_metric():
    table = compute_category_proportions_by_country(
        _make_classified(), min_n=1, n_bootstrap=50, random_state=7
    )
    gaps = compute_fairness_gaps(table)

    assert "bias_alucinacion" in gaps.columns
    assert "bias_correcta" in gaps.columns
    assert "bias_abstencion" in gaps.columns
    gpt = gaps[gaps["model"] == "GPT"].iloc[0]
    assert gpt["bias_alucinacion"] > 0


def test_bootstrap_proportion_bounds():
    labels = np.array(["alucinacion", "abstencion", "alucinacion", "correcta"])
    lower, upper, _ = _bootstrap_proportion(labels, "alucinacion", n_bootstrap=200, random_state=1)
    assert 0 <= lower <= upper <= 100


def test_confiable_flag_when_small_sample():
    df = pd.DataFrame(
        {
            "pais": ["chile"] * 3,
            "model": ["GPT"] * 3,
            "judge_label": ["alucinacion", "abstencion", "correcta"],
        }
    )
    table = compute_category_proportions_by_country(df, min_n=5, n_bootstrap=50)
    assert not table.iloc[0]["confiable"]
    assert "insuficiente" in table.iloc[0]["nota"]


def test_plot_category_proportions_chart(tmp_path):
    table = compute_category_proportions_by_country(
        _make_classified(), min_n=1, n_bootstrap=50, random_state=7
    )
    out = plot_category_proportions_chart(table, tmp_path / "chart.png")
    assert out.exists()


def test_verify_classifications_csv_fresh_rejects_stale_file(tmp_path):
    from analysis.category_proportions_by_country import verify_classifications_csv_fresh

    df = _make_classified()
    path = tmp_path / "low_score_classifications.csv"
    df.to_csv(path, index=False)

    future_start = time.time() + 60
    with pytest.raises(RuntimeError, match="Stale"):
        verify_classifications_csv_fresh(path, df, future_start)


def test_run_proportions_for_current_run_uses_memory(tmp_path):
    import time

    from analysis.category_proportions_by_country import run_proportions_for_current_run

    df = _make_classified()
    csv_path = tmp_path / "low_score_classifications.csv"
    run_started = time.time()
    df.to_csv(csv_path, index=False)

    saved = run_proportions_for_current_run(
        df,
        tmp_path,
        classifications_csv=csv_path,
        run_started_at=run_started,
        min_n=1,
        n_bootstrap=20,
        random_state=1,
    )
    assert (tmp_path / "category_proportions_by_country.csv").exists()
    assert saved["proportions_df"] is not None
