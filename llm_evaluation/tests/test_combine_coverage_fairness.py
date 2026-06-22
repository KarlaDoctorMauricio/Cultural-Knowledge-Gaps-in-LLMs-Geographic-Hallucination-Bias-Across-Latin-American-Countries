import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.combine_coverage_with_fairness import (  # noqa: E402
    build_coverage_vs_fairness_table,
    compute_correlations,
    plot_dispersion_vs_fairness,
    run_combine_analysis,
)


@pytest.fixture
def fake_dirs(tmp_path):
    coverage_dir = tmp_path / "coverage_sample"
    results_dir = tmp_path / "results"
    coverage_dir.mkdir()
    results_dir.mkdir()

    pd.DataFrame(
        {
            "pais": ["chile", "mexico", "peru"],
            "n_preguntas": [15, 15, 15],
            "dispersion_promedio": [1.2, 0.8, 0.5],
            "ranking_cobertura": [1, 2, 3],
        }
    ).to_csv(coverage_dir / "coverage_by_country.csv", index=False)

    pd.DataFrame(
        {
            "pais": ["chile", "mexico", "peru"],
            "model": ["GPT", "GPT", "GPT"],
            "pct_alucinacion": [80.0, 50.0, 30.0],
            "n_total": [10, 10, 10],
        }
    ).to_csv(results_dir / "category_proportions_by_country.csv", index=False)

    pd.DataFrame(
        {
            "pais": ["chile", "mexico", "peru"],
            "method": ["GPT", "GPT", "GPT"],
            "mae": [0.7, 0.5, 0.3],
            "n_preguntas": [10, 10, 10],
        }
    ).to_csv(results_dir / "group_mae_pais.csv", index=False)

    return coverage_dir, results_dir


def test_build_coverage_vs_fairness_table(fake_dirs):
    coverage_dir, results_dir = fake_dirs
    table = build_coverage_vs_fairness_table(coverage_dir, results_dir, model="GPT")

    assert len(table) == 3
    assert set(table.columns) >= {
        "pais",
        "dispersion_promedio",
        "pct_alucinacion",
        "mae",
        "n_preguntas",
    }


def test_compute_correlations(fake_dirs):
    coverage_dir, results_dir = fake_dirs
    table = build_coverage_vs_fairness_table(coverage_dir, results_dir, model="GPT")
    stats = compute_correlations(table)

    assert "dispersion vs pct_alucinacion" in stats
    assert stats["dispersion vs pct_alucinacion"]["spearman_r"] > 0.9


def test_run_combine_analysis_writes_outputs(fake_dirs, tmp_path):
    coverage_dir, results_dir = fake_dirs
    table, correlations, csv_path, plot_path = run_combine_analysis(
        coverage_dir,
        results_dir,
        model="GPT",
        output_csv=tmp_path / "merged.csv",
        output_plot=tmp_path / "plot.png",
    )

    assert csv_path.exists()
    assert plot_path.exists()
    assert len(table) == 3
    assert correlations["dispersion vs pct_alucinacion"]["n"] == 3.0
