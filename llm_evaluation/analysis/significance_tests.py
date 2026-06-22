#!/usr/bin/env python
"""
Post-hoc statistical tests on existing CHOCLO evaluation outputs (read-only).

Outputs:
  - significance_gpt_vs_claude.csv       paired error + global alucinacion tests
  - multiple_comparisons_correction.csv  country-pair proportion tests with FDR
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from results_io import DEFAULT_OUTPUT_DIR  # noqa: E402

SIGNIFICANCE_FILENAME = "significance_gpt_vs_claude.csv"
MULTIPLE_COMP_FILENAME = "multiple_comparisons_correction.csv"

ERROR_COLUMNS = ("error_GPT", "error_Claude")
ALUC_LABEL = "alucinacion"


def _cohens_d_paired(differences: np.ndarray) -> float:
    diff = differences.astype(float)
    std = float(np.std(diff, ddof=1))
    if std == 0:
        return 0.0
    return float(np.mean(diff) / std)


def _two_proportion_pvalue(x1: int, n1: int, x2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return np.nan
    p1 = x1 / n1
    p2 = x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = abs(p1 - p2) / se
    return float(2 * stats.norm.sf(z))


def _mcnemar_pvalue(gpt_aluc_claude_not: int, gpt_not_claude_aluc: int) -> tuple[float, float]:
    """Return (statistic, p-value) for McNemar with continuity correction."""
    b = int(gpt_aluc_claude_not)
    c = int(gpt_not_claude_aluc)
    n = b + c
    if n == 0:
        return np.nan, np.nan
    stat = (abs(b - c) - 1) ** 2 / n
    p_value = float(1 - stats.chi2.cdf(stat, df=1))
    return stat, p_value


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p

    order = np.argsort(p)
    ranked = p[order]
    corrected = np.empty(n, dtype=float)
    for i, pv in enumerate(ranked):
        corrected[i] = min(1.0, pv * n / (i + 1))
    for i in range(n - 2, -1, -1):
        corrected[i] = min(corrected[i], corrected[i + 1])
    out = np.empty(n, dtype=float)
    out[order] = corrected
    return out


def _result_row(
    test_name: str,
    *,
    statistic: float,
    p_value: float,
    effect_size: float,
    effect_size_type: str,
    n: int,
    notes: str = "",
    extra: dict | None = None,
) -> dict:
    row = {
        "test_name": test_name,
        "statistic": round(statistic, 6) if pd.notna(statistic) else np.nan,
        "p_value": round(p_value, 6) if pd.notna(p_value) else np.nan,
        "effect_size": round(effect_size, 6) if pd.notna(effect_size) else np.nan,
        "effect_size_type": effect_size_type,
        "n": n,
        "notes": notes,
    }
    if extra:
        row.update(extra)
    return row


def run_paired_error_tests(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon + Shapiro + paired t-test on error_GPT vs error_Claude."""
    missing = [c for c in ERROR_COLUMNS if c not in evaluation_df.columns]
    if missing:
        raise ValueError(f"evaluation_results.csv missing columns: {missing}")

    paired = evaluation_df[list(ERROR_COLUMNS)].dropna()
    gpt = paired["error_GPT"].to_numpy(dtype=float)
    claude = paired["error_Claude"].to_numpy(dtype=float)
    diff = gpt - claude
    n = len(diff)

    wilcoxon = stats.wilcoxon(gpt, claude, alternative="two-sided", method="auto")
    shapiro = stats.shapiro(diff) if 3 <= n <= 5000 else (np.nan, np.nan)
    ttest = stats.ttest_rel(gpt, claude, nan_policy="omit")

    median_diff = float(np.median(diff))
    mean_diff = float(np.mean(diff))
    cohens_d = _cohens_d_paired(diff)

    rows = [
        _result_row(
            "wilcoxon_signed_rank_error",
            statistic=float(wilcoxon.statistic),
            p_value=float(wilcoxon.pvalue),
            effect_size=median_diff,
            effect_size_type="median_diff_error_GPT_minus_Claude",
            n=n,
            notes="Primary test: non-parametric paired comparison on 1-similarity.",
        ),
        _result_row(
            "shapiro_wilk_difference",
            statistic=float(shapiro[0]) if pd.notna(shapiro[0]) else np.nan,
            p_value=float(shapiro[1]) if pd.notna(shapiro[1]) else np.nan,
            effect_size=np.nan,
            effect_size_type="normality_check_on_diff",
            n=n,
            notes="H0: differences are normal. Low p => prefer Wilcoxon over t-test.",
        ),
        _result_row(
            "paired_t_test_error",
            statistic=float(ttest.statistic),
            p_value=float(ttest.pvalue),
            effect_size=cohens_d,
            effect_size_type="cohens_d_paired",
            n=n,
            notes=f"mean_diff={mean_diff:.4f}; use only if Shapiro is not significant.",
        ),
    ]
    return pd.DataFrame(rows)


def _paired_alucinacion_table(classifications: pd.DataFrame) -> pd.DataFrame:
    gpt = classifications[classifications["model"] == "GPT"].set_index("row_index")
    claude = classifications[classifications["model"] == "Claude"].set_index("row_index")
    shared = gpt.index.intersection(claude.index)

    rows = []
    for idx in shared:
        g_label = str(gpt.at[idx, "judge_label"])
        c_label = str(claude.at[idx, "judge_label"])
        rows.append(
            {
                "row_index": idx,
                "gpt_aluc": g_label == ALUC_LABEL,
                "claude_aluc": c_label == ALUC_LABEL,
            }
        )
    return pd.DataFrame(rows)


def run_global_alucinacion_tests(classifications: pd.DataFrame) -> pd.DataFrame:
    """McNemar (paired) + two-sample proportion z-test (independent framing)."""
    paired = _paired_alucinacion_table(classifications)
    n = len(paired)

    gpt_aluc = int(paired["gpt_aluc"].sum())
    claude_aluc = int(paired["claude_aluc"].sum())
    gpt_rate = 100.0 * gpt_aluc / n
    claude_rate = 100.0 * claude_aluc / n

    both_no = int((~paired["gpt_aluc"] & ~paired["claude_aluc"]).sum())
    both_yes = int((paired["gpt_aluc"] & paired["claude_aluc"]).sum())
    gpt_only = int((paired["gpt_aluc"] & ~paired["claude_aluc"]).sum())
    claude_only = int((~paired["gpt_aluc"] & paired["claude_aluc"]).sum())

    mcnemar_stat, mcnemar_p = _mcnemar_pvalue(gpt_only, claude_only)
    prop_p = _two_proportion_pvalue(gpt_aluc, n, claude_aluc, n)

    risk_diff = (gpt_aluc / n) - (claude_aluc / n)

    rows = [
        _result_row(
            "mcnemar_alucinacion_paired",
            statistic=mcnemar_stat,
            p_value=mcnemar_p,
            effect_size=risk_diff,
            effect_size_type="risk_diff_GPT_minus_Claude",
            n=n,
            notes=(
                f"GPT {gpt_rate:.1f}% ({gpt_aluc}/{n}) vs Claude {claude_rate:.1f}% "
                f"({claude_aluc}/{n}); discordant GPT-only={gpt_only}, Claude-only={claude_only}; "
                f"both_aluc={both_yes}, neither={both_no}. Primary paired test."
            ),
        ),
        _result_row(
            "two_proportion_z_alucinacion_independent",
            statistic=np.nan,
            p_value=prop_p,
            effect_size=risk_diff,
            effect_size_type="risk_diff_GPT_minus_Claude",
            n=n,
            notes="Supplementary: treats the two model rates as independent (less appropriate).",
        ),
    ]
    return pd.DataFrame(rows)


def ci_overlap(
    low_a: float,
    high_a: float,
    low_b: float,
    high_b: float,
) -> bool:
    return not (high_a < low_b or high_b < low_a)


def _highlight_pairs_from_gaps(fairness_gaps: pd.DataFrame) -> set[tuple[str, str, str]]:
    """Return {(model, pais_a, pais_b), ...} for max vs min alucinacion per model."""
    pairs: set[tuple[str, str, str]] = set()
    for _, row in fairness_gaps.iterrows():
        model = str(row["model"])
        a = str(row.get("max_pais_alucinacion", ""))
        b = str(row.get("min_pais_alucinacion", ""))
        if a and b and a != b:
            key = tuple(sorted((a, b)))
            pairs.add((model, key[0], key[1]))
    return pairs


def run_multiple_comparisons(
    proportions: pd.DataFrame,
    fairness_gaps: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Country-pair proportion tests with CI overlap flags and BH / Bonferroni correction.

    Corrections are applied to the union of:
      - pairs with non-overlapping bootstrap CIs on pct_alucinacion
      - max vs min country pairs from fairness_gaps (if provided)
    """
    required = {
        "pais",
        "model",
        "n_alucinacion",
        "n_total",
        "pct_alucinacion_ci_lower",
        "pct_alucinacion_ci_upper",
    }
    missing = required - set(proportions.columns)
    if missing:
        raise ValueError(f"category_proportions_by_country.csv missing: {sorted(missing)}")

    highlights = _highlight_pairs_from_gaps(fairness_gaps) if fairness_gaps is not None else set()
    rows: list[dict] = []

    for model, model_df in proportions.groupby("model", sort=True):
        countries = model_df.set_index("pais")
        for pais_a, pais_b in combinations(sorted(countries.index.astype(str)), 2):
            a = countries.loc[pais_a]
            b = countries.loc[pais_b]

            x1 = int(a["n_alucinacion"])
            n1 = int(a["n_total"])
            x2 = int(b["n_alucinacion"])
            n2 = int(b["n_total"])

            p_raw = _two_proportion_pvalue(x1, n1, x2, n2)
            overlap = ci_overlap(
                float(a["pct_alucinacion_ci_lower"]),
                float(a["pct_alucinacion_ci_upper"]),
                float(b["pct_alucinacion_ci_lower"]),
                float(b["pct_alucinacion_ci_upper"]),
            )
            is_highlight = (model, pais_a, pais_b) in highlights or (
                model,
                pais_b,
                pais_a,
            ) in highlights

            rows.append(
                {
                    "model": model,
                    "comparacion": f"{pais_a} vs {pais_b}",
                    "pais_a": pais_a,
                    "pais_b": pais_b,
                    "pct_a": float(a["pct_alucinacion"]),
                    "pct_b": float(b["pct_alucinacion"]),
                    "ci_overlap": overlap,
                    "fairness_gap_pair": is_highlight,
                    "p_value_raw": p_raw,
                }
            )

    all_pairs = pd.DataFrame(rows)
    if all_pairs.empty:
        return all_pairs

    report_mask = (~all_pairs["ci_overlap"]) | all_pairs["fairness_gap_pair"]
    report = all_pairs[report_mask].copy().reset_index(drop=True)

    if report.empty:
        report = all_pairs.copy()

    p_raw = report["p_value_raw"].to_numpy(dtype=float)
    report["p_value_corrected_bh"] = benjamini_hochberg(p_raw)
    report["p_value_corrected_bonferroni"] = np.minimum(p_raw * len(report), 1.0)
    report["p_value_corrected"] = report["p_value_corrected_bh"]
    report["correction_method"] = "benjamini_hochberg"
    report["significativo_post_correccion"] = report["p_value_corrected"] < 0.05
    report["significativo_bonferroni"] = report["p_value_corrected_bonferroni"] < 0.05

    report = report.sort_values(
        ["model", "p_value_raw"],
        ascending=[True, True],
        kind="stable",
    ).reset_index(drop=True)

    for col in ("p_value_raw", "p_value_corrected", "p_value_corrected_bh", "p_value_corrected_bonferroni"):
        report[col] = report[col].round(6)

    return report


def run_all_significance_tests(
    results_dir: Path,
    *,
    alpha: float = 0.05,
) -> dict[str, pd.DataFrame]:
    results_dir = Path(results_dir)
    eval_path = results_dir / "evaluation_results.csv"
    class_path = results_dir / "low_score_classifications.csv"
    prop_path = results_dir / "category_proportions_by_country.csv"
    gaps_path = results_dir / "category_proportions_fairness_gaps.csv"

    evaluation_df = pd.read_csv(eval_path)
    error_tests = run_paired_error_tests(evaluation_df)

    aluc_tests = pd.DataFrame()
    if class_path.exists():
        classifications = pd.read_csv(class_path)
        aluc_tests = run_global_alucinacion_tests(classifications)

    significance = pd.concat([error_tests, aluc_tests], ignore_index=True)
    significance["alpha"] = alpha

    multiple = pd.DataFrame()
    if prop_path.exists():
        proportions = pd.read_csv(prop_path)
        gaps = pd.read_csv(gaps_path) if gaps_path.exists() else None
        multiple = run_multiple_comparisons(proportions, gaps)

    results_dir.mkdir(parents=True, exist_ok=True)
    significance.to_csv(results_dir / SIGNIFICANCE_FILENAME, index=False, encoding="utf-8")
    if not multiple.empty:
        multiple.to_csv(results_dir / MULTIPLE_COMP_FILENAME, index=False, encoding="utf-8")

    return {
        SIGNIFICANCE_FILENAME: significance,
        MULTIPLE_COMP_FILENAME: multiple,
    }


def print_summary(outputs: dict[str, pd.DataFrame]) -> None:
    sig = outputs.get(SIGNIFICANCE_FILENAME, pd.DataFrame())
    mult = outputs.get(MULTIPLE_COMP_FILENAME, pd.DataFrame())

    if not sig.empty:
        print("\n--- GPT vs Claude: tests de significancia ---")
        for _, row in sig.iterrows():
            print(
                f"  {row['test_name']}: stat={row['statistic']}, p={row['p_value']}, "
                f"effect ({row['effect_size_type']})={row['effect_size']} "
                f"[n={row['n']}]"
            )
            if row.get("notes"):
                print(f"    {row['notes']}")

    if not mult.empty:
        n_no_overlap = int((~mult["ci_overlap"]).sum()) if "ci_overlap" in mult.columns else 0
        print("\n--- Comparaciones múltiples (BH, pares sin traslape de CI o max vs min) ---")
        print(
            f"  Pares en tabla de corrección: {len(mult)} "
            f"(sin traslape de CI: {n_no_overlap}; incluye max vs min de fairness_gaps)"
        )
        sig_after = mult[mult["significativo_post_correccion"]]
        print(f"  Significativos post-BH (alpha=0.05): {len(sig_after)}")
        for model in mult["model"].unique():
            subset = mult[(mult["model"] == model) & mult["fairness_gap_pair"]]
            if not subset.empty:
                row = subset.iloc[0]
                print(
                    f"  {model} gap pair ({row['comparacion']}): "
                    f"p_raw={row['p_value_raw']}, p_BH={row['p_value_corrected']}, "
                    f"CI_overlap={row['ci_overlap']}, "
                    f"sig={row['significativo_post_correccion']}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-hoc significance tests on saved CHOCLO results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory with evaluation_results.csv and related outputs.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_all_significance_tests(args.results_dir, alpha=args.alpha)
    print_summary(outputs)
    print(f"\nSaved to {args.results_dir.resolve()}:")
    for name, df in outputs.items():
        if df is not None and not df.empty:
            print(f"  - {name} ({len(df)} rows)")


if __name__ == "__main__":
    main()
