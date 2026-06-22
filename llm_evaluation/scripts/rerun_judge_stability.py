#!/usr/bin/env python
"""
Re-run the LLM judge on existing evaluation_results.csv (3 stability passes).

Does NOT regenerate GPT/Claude answers. Archives pre-fix judge outputs, runs
three independent classifications with the corrected judge system prompt, and
writes consolidated fairness tables from the modal label across runs.

Example:
    python rerun_judge_stability.py
    python rerun_judge_stability.py --runs 3 --skip-archive
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.category_proportions_by_country import (  # noqa: E402
    compute_category_proportions_by_country,
    compute_fairness_gaps,
    plot_category_proportions_chart,
)
from evaluator import (  # noqa: E402
    JUDGE_MODELS,
    JUDGE_PROMPT_VERSION,
    QUALITY_LABELS,
    build_response_quality_breakdown_table,
    build_response_quality_by_country_table,
    classify_low_score_responses,
)
from results_io import DEFAULT_OUTPUT_DIR  # noqa: E402

ARCHIVE_DIRNAME = "archive_pre_fix"
ARCHIVE_FILES = (
    "response_quality_breakdown.csv",
    "category_proportions_by_country.csv",
    "category_proportions_fairness_gaps.csv",
    "hallucination_by_country.csv",
    "low_score_classifications.csv",
)
PCT_COLUMNS = tuple(f"pct_{label}" for label in QUALITY_LABELS)


def archive_pre_fix_outputs(output_dir: Path) -> list[Path]:
    """Move pre-fix judge CSVs into archive_pre_fix/ (do not delete)."""
    archive_dir = output_dir / ARCHIVE_DIRNAME
    archive_dir.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []

    for name in ARCHIVE_FILES:
        src = output_dir / name
        if not src.exists():
            continue
        dest = archive_dir / name
        if dest.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            dest = archive_dir / f"{src.stem}_{stamp}{src.suffix}"
        shutil.move(str(src), str(dest))
        moved.append(dest)

    return moved


def _mode_label(labels: pd.Series) -> str:
    counts = labels.value_counts()
    if counts.empty:
        return str(labels.iloc[0])
    top = counts[counts == counts.max()].index.tolist()
    return sorted(top)[0]


def build_judge_stability_table(runs: list[pd.DataFrame]) -> tuple[pd.DataFrame, float]:
    """Compare judge labels across runs; return stability table and pct stable."""
    if len(runs) < 2:
        raise ValueError("Need at least two judge runs for stability analysis.")

    merged = runs[0][["row_index", "model", "pais", "judge_label"]].rename(
        columns={"judge_label": "label_run1"}
    )
    for idx, run_df in enumerate(runs[1:], start=2):
        merged = merged.merge(
            run_df[["row_index", "model", "judge_label"]].rename(
                columns={"judge_label": f"label_run{idx}"}
            ),
            on=["row_index", "model"],
            how="inner",
        )

    label_cols = [f"label_run{i}" for i in range(1, len(runs) + 1)]
    merged["stable"] = merged[label_cols].nunique(axis=1) == 1
    merged["final_label"] = merged[label_cols].apply(_mode_label, axis=1)

    def _variation(row: pd.Series) -> str:
        if row["stable"]:
            return row["final_label"]
        parts = [f"run{i}={row[col]}" for i, col in enumerate(label_cols, start=1)]
        return "; ".join(parts)

    merged["variation_detail"] = merged.apply(_variation, axis=1)
    pct_stable = float(merged["stable"].mean() * 100.0)
    return merged, pct_stable


def consolidate_judge_runs(
    runs: list[pd.DataFrame],
    *,
    judge_prompt_version: str = JUDGE_PROMPT_VERSION,
) -> pd.DataFrame:
    """Build judge_final_results.csv using modal label; keep full metadata from run 1."""
    stability, _ = build_judge_stability_table(runs)
    base = runs[0].copy()
    labels = stability[["row_index", "model", "final_label", "stable"]]
    base = base.drop(columns=["judge_label"], errors="ignore")
    base = base.merge(labels, on=["row_index", "model"], how="left")
    base = base.rename(columns={"final_label": "judge_label", "stable": "judge_stability"})
    base["judge_prompt_version"] = judge_prompt_version
    base["judge_consolidation"] = f"mode_of_{len(runs)}_runs"
    return base


def build_before_after_prompt_fix(
    archive_dir: Path,
    new_breakdown: pd.DataFrame,
) -> pd.DataFrame:
    """Compare archived response_quality_breakdown vs post-fix consolidated breakdown."""
    old_path = archive_dir / "response_quality_breakdown.csv"
    if not old_path.exists():
        return pd.DataFrame()

    old = pd.read_csv(old_path)
    merged = old.merge(
        new_breakdown,
        on="model",
        how="outer",
        suffixes=("_before", "_after"),
    )

    rows: list[dict] = []
    for _, row in merged.iterrows():
        entry = {
            "model": row["model"],
            "judge_prompt_before": "v1_generic_system_prompt",
            "judge_prompt_after": JUDGE_PROMPT_VERSION,
        }
        for col in PCT_COLUMNS:
            before = row.get(f"{col}_before", np.nan)
            after = row.get(f"{col}_after", np.nan)
            entry[f"{col}_before"] = before
            entry[f"{col}_after"] = after
            if pd.notna(before) and pd.notna(after):
                entry[f"delta_{col.removeprefix('pct_')}"] = round(float(after - before), 2)
            else:
                entry[f"delta_{col.removeprefix('pct_')}"] = np.nan
        rows.append(entry)

    return pd.DataFrame(rows)


def update_run_summary_judge_rerun(
    output_dir: Path,
    *,
    pct_filas_estables: float,
    n_runs: int,
    judge_prompt_version: str,
) -> Path:
    """Append judge rerun metadata without erasing the original evaluation summary."""
    summary_path = output_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if "evaluation_generated_at" not in summary and "generated_at" in summary:
            summary["evaluation_generated_at"] = summary["generated_at"]
    else:
        summary = {}

    now = datetime.now(timezone.utc).isoformat()
    summary["judge_rerun"] = {
        "generated_at": now,
        "judge_prompt_version": judge_prompt_version,
        "n_runs": n_runs,
        "pct_filas_estables": round(pct_filas_estables, 2),
        "source_of_truth": "judge_final_results.csv",
        "archived_pre_fix_dir": str((output_dir / ARCHIVE_DIRNAME).resolve()),
    }
    summary["judge_generated_at"] = now
    summary["judge_prompt_version"] = judge_prompt_version
    summary["judge_source_file"] = "judge_final_results.csv"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


def save_run_outputs(
    run_df: pd.DataFrame,
    path: Path,
    *,
    run_id: int,
    judge_prompt_version: str,
) -> None:
    slim = run_df[["row_index", "pais", "model", "judge_label"]].copy()
    slim["judge_prompt_version"] = judge_prompt_version
    slim["judge_run_id"] = run_id
    slim.to_csv(path, index=False, encoding="utf-8")


def rerun_judge_stability(
    output_dir: Path,
    *,
    evaluation_path: Path | None = None,
    n_runs: int = 3,
    skip_archive: bool = False,
    show_progress: bool = True,
    min_group_n: int = 5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    save_chart: bool = True,
) -> dict[str, Path | pd.DataFrame | float]:
    output_dir = Path(output_dir)
    evaluation_path = evaluation_path or output_dir / "evaluation_results.csv"
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Missing {evaluation_path}")

    moved: list[Path] = []
    if not skip_archive:
        moved = archive_pre_fix_outputs(output_dir)
        print(f"Archived {len(moved)} pre-fix file(s) to {output_dir / ARCHIVE_DIRNAME}")

    scored_df = pd.read_csv(evaluation_path)
    runs: list[pd.DataFrame] = []

    for run_id in range(1, n_runs + 1):
        print(f"\n--- Judge run {run_id}/{n_runs} ({JUDGE_PROMPT_VERSION}) ---")
        started = time.time()
        classified = classify_low_score_responses(
            scored_df,
            model_names=JUDGE_MODELS,
            checkpoint_dir=None,
            resume=False,
            show_progress=show_progress,
            judge_prompt_version=JUDGE_PROMPT_VERSION,
        )
        elapsed = time.time() - started
        run_path = output_dir / f"judge_run{run_id}.csv"
        save_run_outputs(
            classified,
            run_path,
            run_id=run_id,
            judge_prompt_version=JUDGE_PROMPT_VERSION,
        )
        runs.append(classified)
        print(f"      Saved {len(classified):,} rows to {run_path.name} ({elapsed/60:.1f} min)")

    stability_df, pct_stable = build_judge_stability_table(runs)
    stability_path = output_dir / "judge_stability.csv"
    stability_df.to_csv(stability_path, index=False, encoding="utf-8")

    final_df = consolidate_judge_runs(runs)
    final_path = output_dir / "judge_final_results.csv"
    final_df.to_csv(final_path, index=False, encoding="utf-8")

    breakdown = build_response_quality_breakdown_table(final_df)
    breakdown["judge_prompt_version"] = JUDGE_PROMPT_VERSION
    breakdown["judge_source_file"] = final_path.name
    breakdown_path = output_dir / "response_quality_breakdown.csv"
    breakdown.to_csv(breakdown_path, index=False, encoding="utf-8")

    by_country = build_response_quality_by_country_table(final_df)
    by_country["judge_prompt_version"] = JUDGE_PROMPT_VERSION
    by_country_path = output_dir / "hallucination_by_country.csv"
    by_country.to_csv(by_country_path, index=False, encoding="utf-8")

    proportions = compute_category_proportions_by_country(
        final_df,
        min_n=min_group_n,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    proportions["judge_prompt_version"] = JUDGE_PROMPT_VERSION
    proportions["judge_source_file"] = final_path.name
    proportions_path = output_dir / "category_proportions_by_country.csv"
    proportions.to_csv(proportions_path, index=False, encoding="utf-8")

    gaps = compute_fairness_gaps(proportions)
    gaps["judge_prompt_version"] = JUDGE_PROMPT_VERSION
    gaps["judge_source_file"] = final_path.name
    gaps_path = output_dir / "category_proportions_fairness_gaps.csv"
    gaps.to_csv(gaps_path, index=False, encoding="utf-8")

    if save_chart and not proportions.empty:
        plot_category_proportions_chart(
            proportions,
            output_dir / "category_proportions_chart.png",
        )

    final_df.to_csv(output_dir / "low_score_classifications.csv", index=False, encoding="utf-8")

    archive_dir = output_dir / ARCHIVE_DIRNAME
    before_after = build_before_after_prompt_fix(archive_dir, breakdown)
    before_after_path = output_dir / "before_after_prompt_fix.csv"
    if not before_after.empty:
        before_after.to_csv(before_after_path, index=False, encoding="utf-8")

    summary_path = update_run_summary_judge_rerun(
        output_dir,
        pct_filas_estables=pct_stable,
        n_runs=n_runs,
        judge_prompt_version=JUDGE_PROMPT_VERSION,
    )

    print(f"\nEstabilidad: {pct_stable:.1f}% filas con la misma etiqueta en {n_runs} corridas")
    unstable = stability_df[~stability_df["stable"]]
    if not unstable.empty:
        print(f"  Filas inestables: {len(unstable):,} (ver variation_detail en judge_stability.csv)")

    return {
        "judge_stability.csv": stability_df,
        "judge_final_results.csv": final_df,
        "before_after_prompt_fix.csv": before_after,
        "pct_filas_estables": pct_stable,
        "run_summary.json": summary_path,
        "stability_path": stability_path,
        "final_path": final_path,
        "before_after_path": before_after_path if not before_after.empty else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run CHOCLO judge 3x for stability (no response regeneration)."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory with evaluation_results.csv and judge outputs.",
    )
    parser.add_argument(
        "--evaluation-results",
        type=Path,
        default=None,
        help="Path to evaluation_results.csv (default: <results-dir>/evaluation_results.csv).",
    )
    parser.add_argument("--runs", type=int, default=3, help="Independent judge passes (default: 3).")
    parser.add_argument(
        "--skip-archive",
        action="store_true",
        help="Do not move existing judge CSVs to archive_pre_fix/.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_path = args.evaluation_results or args.results_dir / "evaluation_results.csv"
    outputs = rerun_judge_stability(
        args.results_dir,
        evaluation_path=eval_path,
        n_runs=args.runs,
        skip_archive=args.skip_archive,
        show_progress=not args.quiet,
    )

    print(f"\nOutputs in {args.results_dir.resolve()}:")
    for name in (
        "judge_run1.csv",
        "judge_run2.csv",
        "judge_run3.csv",
        "judge_stability.csv",
        "judge_final_results.csv",
        "before_after_prompt_fix.csv",
        "response_quality_breakdown.csv",
        "category_proportions_fairness_gaps.csv",
        "run_summary.json",
    ):
        path = args.results_dir / name
        if path.exists():
            print(f"  - {name}")
    print(f"  pct_filas_estables = {outputs['pct_filas_estables']:.2f}%")


if __name__ == "__main__":
    main()
