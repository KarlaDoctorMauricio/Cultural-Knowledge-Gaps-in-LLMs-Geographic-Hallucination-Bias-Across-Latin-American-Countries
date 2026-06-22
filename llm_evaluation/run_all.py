#!/usr/bin/env python
"""
Run the full CHOCLO evaluation workflow in one command (no plots).

Steps:
  1. Build stratified sample (skipped if data/choclo_sample.csv already exists)
  2. Evaluate GPT/Claude (LatamGPT off by default), fairness metrics, judge analysis
  3. Save results + category composition analysis
  4. Export low-score examples and manual review text
  5. Primary fairness: judge proportions by country (in-memory, verified fresh)
  6. Report tables: IR, alucinacion_country_full, UMAP (unless --skip-report)

Does NOT run rerun_judge_stability.py nor temperature_sensitivity.py by default.
After changing the judge prompt, run: python scripts/rerun_judge_stability.py
Then refresh report tables: python run_report.py

Example:
    python run_all.py
    python run_all.py -n 30
    python run_all.py --skip-report
    python run_report.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairness_toolkit.choclo import (  # noqa: E402
    CHOCLO_HF_CSV,
    LOCAL_SAMPLE_PATH,
    load_choclo,
    sample_choclo,
)
from fairness_toolkit.env import load_env  # noqa: E402
from fairness_toolkit.checkpoint import default_checkpoint_dir  # noqa: E402
from fairness_toolkit.pipeline import DEFAULT_RUN_MODELS, run_choclo_evaluation  # noqa: E402
from analysis.inspect_low_scores import (  # noqa: E402
    DEFAULT_COUNTRIES,
    REVIEW_COUNTRIES,
    REVIEW_MAX_RANK,
    TOP_K,
    build_manual_review,
    export_manual_review,
    extract_low_scores,
)
from results_io import DEFAULT_OUTPUT_DIR, save_results  # noqa: E402
from analysis.analyze_category_composition import (  # noqa: E402
    run_category_composition_analysis,
    save_analysis_outputs,
    print_analysis_summary,
)
from analysis.category_proportions_by_country import (  # noqa: E402
    clear_stale_proportion_outputs,
    print_fairness_summary,
    run_proportions_for_current_run,
)
from analysis.analyze_alucinacion_composition import (  # noqa: E402
    print_alucinacion_composition_summary,
    run_alucinacion_composition_analysis,
)
from run_report import run_report_analyses  # noqa: E402
from scripts.sample_choclo import build_stratified_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sample + evaluation + low-score review in one shot."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=LOCAL_SAMPLE_PATH,
        help=f"Input CHOCLO CSV (default: {LOCAL_SAMPLE_PATH}).",
    )
    parser.add_argument(
        "--refresh-sample",
        action="store_true",
        help="Rebuild the stratified sample even if --data already exists.",
    )
    parser.add_argument(
        "--per-country",
        type=int,
        default=15,
        help="Questions per country when building the sample (default: 15).",
    )
    parser.add_argument(
        "--sample-source",
        type=str,
        default=CHOCLO_HF_CSV,
        help="Source for --refresh-sample (default: Hugging Face CHOCLO CSV).",
    )
    parser.add_argument(
        "-n",
        "--sample-n",
        type=int,
        default=None,
        help="Random subset size for a quick test (e.g. -n 30).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Results directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="CHOCLO",
        help="Experiment label (default: CHOCLO).",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip group fairness post-processing calibration.",
    )
    parser.add_argument(
        "--min-group-n",
        type=int,
        default=5,
        help="Minimum questions per group for confiable flag (default: 5).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples for group MAE CIs (default: 1000).",
    )
    parser.add_argument(
        "--inspect-countries",
        nargs="+",
        default=list(DEFAULT_COUNTRIES),
        help="Countries for low-score export (default: argentina chile panama).",
    )
    parser.add_argument(
        "--inspect-top-k",
        type=int,
        default=TOP_K,
        help="Lowest-score examples per country/model (default: 5).",
    )
    parser.add_argument(
        "--print-review",
        action="store_true",
        help="Print full manual review text to the console.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save partial progress every N API tasks (default: 10).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Checkpoint directory (default: <output-dir>/checkpoints).",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable incremental checkpoint/resume.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoint files and start fresh.",
    )
    parser.add_argument(
        "--include-latamgpt",
        action="store_true",
        help="Include LatamGPT via Hugging Face router (default: GPT + Claude only).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress bars (phase logs still print).",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip report tables (IR, country full, UMAP) at end of run.",
    )
    parser.add_argument(
        "--skip-coverage",
        action="store_true",
        help="With report: skip UMAP and coverage_vs_fairness (faster).",
    )
    return parser.parse_args()


def ensure_sample(args: argparse.Namespace) -> Path:
    data_path = args.data
    if data_path.exists() and not args.refresh_sample:
        print(f"[1/5] Using existing sample: {data_path}")
        return data_path

    print(f"[1/5] Building stratified sample ({args.per_country} per country) ...")
    df = load_choclo(args.sample_source)
    sample = build_stratified_sample(
        df,
        n_per_country=args.per_country,
        random_state=args.random_state,
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(data_path, index=False, encoding="utf-8")
    print(f"      Saved {len(sample):,} rows to {data_path}")
    return data_path


def run_inspection(
    evaluation_csv: Path,
    output_dir: Path,
    countries: tuple[str, ...],
    top_k: int,
    print_review: bool,
) -> dict[str, Path]:
    print("[4/5] Exporting low-score examples and manual review ...")
    df = pd.read_csv(evaluation_csv)

    examples = extract_low_scores(
        df,
        countries=countries,
        top_k=top_k,
    )

    examples_path = output_dir / "low_scores_examples.csv"
    review_path = output_dir / "low_scores_review.txt"

    examples.to_csv(examples_path, index=False, encoding="utf-8")
    review_subset = export_manual_review(examples, review_path)

    print(f"      Saved {len(examples):,} examples to {examples_path}")
    print(
        f"      Saved manual review ({len(review_subset):,} cases, "
        f"{', '.join(REVIEW_COUNTRIES)} rank 1-{REVIEW_MAX_RANK}) to {review_path}"
    )

    if print_review:
        print("\n" + build_manual_review(examples))

    return {
        "low_scores_examples.csv": examples_path,
        "low_scores_review.txt": review_path,
    }


def main() -> None:
    args = parse_args()
    load_env()

    print("=" * 60)
    print("CHOCLO full pipeline (sample + eval + review, no plots)")
    print("=" * 60)

    data_path = ensure_sample(args)

    model_names = list(DEFAULT_RUN_MODELS)
    if args.include_latamgpt:
        model_names = ["LatamGPT", *model_names]

    print(f"[2/5] Evaluating questions from {data_path} ...")
    df = load_choclo(data_path)
    if args.sample_n is not None:
        df = sample_choclo(df, n=args.sample_n, random_state=args.random_state)
    print(f"      {len(df):,} questions")
    print(f"      Modelos: {', '.join(model_names)}")
    if not args.include_latamgpt:
        print("      (LatamGPT excluido; usar --include-latamgpt para incluirlo)")

    api_calls = len(df) * len(model_names)
    judge_calls = len(df) * len([m for m in ("GPT", "Claude") if m in model_names])
    print(
        f"      Estimado: ~{api_calls:,} llamadas respuesta + "
        f"~{judge_calls:,} llamadas judge"
    )

    checkpoint_dir = None
    if not args.no_checkpoint:
        checkpoint_dir = args.checkpoint_dir or default_checkpoint_dir(args.output_dir)
        print(f"      Checkpoints: {checkpoint_dir.resolve()} (every {args.checkpoint_every})")

    run_started_at = time.time()
    clear_stale_proportion_outputs(args.output_dir)

    results = run_choclo_evaluation(
        df,
        config_name=args.config_name,
        model_names=model_names,
        apply_calibration=not args.no_calibration,
        min_group_n=args.min_group_n,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
        show_progress=not args.quiet,
    )

    print("[3/5] Saving results and category composition ...")

    saved = save_results(results, args.output_dir)
    composition_outputs = run_category_composition_analysis(
        results["scored_df"],
        results["df_group_mae_pais"],
        results["df_group_mae_category"],
        results.get("df_group_mae_interseccion"),
    )
    composition_saved = save_analysis_outputs(composition_outputs, args.output_dir)
    saved.update(composition_saved)
    print_analysis_summary(composition_outputs)

    inspection_saved = run_inspection(
        args.output_dir / "evaluation_results.csv",
        args.output_dir,
        countries=tuple(args.inspect_countries),
        top_k=args.inspect_top_k,
        print_review=args.print_review,
    )
    saved.update(inspection_saved)

    classifications = results.get("low_score_classifications")
    proportions_saved: dict = {}
    classifications_csv = args.output_dir / "low_score_classifications.csv"
    if classifications is not None and not classifications.empty:
        print(
            f"[5/5] Primary fairness proportions from this run "
            f"({len(classifications):,} in-memory classifications, not reading stale CSV) ..."
        )
        proportions_saved = run_proportions_for_current_run(
            classifications,
            args.output_dir,
            classifications_csv=classifications_csv,
            run_started_at=run_started_at,
            min_n=args.min_group_n,
            n_bootstrap=args.n_bootstrap,
            random_state=args.random_state,
        )
        for key, value in proportions_saved.items():
            if isinstance(value, Path):
                saved[key] = value
        print_fairness_summary(proportions_saved["fairness_gaps_df"])

        alucinacion_comp = run_alucinacion_composition_analysis(
            classifications,
            proportions_saved["proportions_df"],
            results["scored_df"],
            args.output_dir,
        )
        print_alucinacion_composition_summary(alucinacion_comp)
        for filename, df in alucinacion_comp.items():
            if df is not None and not df.empty:
                saved[filename] = args.output_dir / filename
    else:
        print("[5/5] Skipping proportion analysis (no classifications in this run).")

    if not args.skip_report and classifications is not None and not classifications.empty:
        print("[6/6] Report tables (IR, country full, UMAP) ...")
        report_saved = run_report_analyses(
            args.output_dir,
            refresh_judge_tables=False,
            run_coverage=not args.skip_coverage,
            min_n=args.min_group_n,
            n_bootstrap=args.n_bootstrap,
            random_state=args.random_state,
        )
        saved.update(report_saved)
    elif args.skip_report:
        print("[6/6] Skipped report (--skip-report). Run: python run_report.py")

    print("\n" + "=" * 60)
    print("Done.")
    print(f"Best model (MAE secondary): {results['best_method']}")
    if classifications is not None and not classifications.empty:
        gaps = proportions_saved.get("fairness_gaps_df")
        if gaps is not None and not gaps.empty:
            primary = gaps.sort_values("bias_alucinacion").iloc[0]
            print(
                f"Primary fairness (lowest hallucination gap): {primary['model']} "
                f"(bias_alucinacion={primary['bias_alucinacion']:.2f} pp)"
            )
    print(f"Available models: {results['available_models']}")
    if results.get("unavailable_models"):
        print(f"Unavailable models: {results['unavailable_models']}")
    if results.get("latamgpt_note"):
        print(f"LatamGPT note: {results['latamgpt_note']}")
    print(f"\nAll outputs in: {args.output_dir.resolve()}")
    for name in sorted(saved):
        print(f"  - {name}")
    if args.skip_report:
        print("\nPara tablas del informe: python run_report.py")


if __name__ == "__main__":
    main()
