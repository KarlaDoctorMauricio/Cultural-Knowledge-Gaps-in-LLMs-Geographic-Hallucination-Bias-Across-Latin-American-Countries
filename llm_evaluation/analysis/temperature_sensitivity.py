#!/usr/bin/env python
"""
Temperature sensitivity experiment for CHOCLO response generation.

Distinct from generation_stability_analysis.py: here the axis is temperature
(0.4 and 0.6), not repetition. Each question-model-temperature pair is generated
and judged exactly once. Baseline temperature=0.2 is reused from the main
evaluation (evaluation_results.csv / response_quality_breakdown.csv).

Example:
    python analysis/temperature_sensitivity.py --sample-n 10
    python analysis/temperature_sensitivity.py
    python analysis/temperature_sensitivity.py --no-resume
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clients import query_claude, query_gpt  # noqa: E402
from evaluator import (  # noqa: E402
    JUDGE_PROMPT_VERSION,
    QUALITY_LABELS,
    SemanticSimilarityScorer,
    llm_judge_classify,
    score_embedding,
)
from fairness_toolkit.choclo import LOCAL_SAMPLE_PATH, add_group_columns, load_choclo  # noqa: E402
from fairness_toolkit.env import load_env  # noqa: E402
from fairness_toolkit.progress import log_phase, log_step, task_progress  # noqa: E402
from results_io import DEFAULT_OUTPUT_DIR  # noqa: E402

MODELS = ("GPT", "Claude")
NEW_TEMPERATURES = (0.4, 0.6)
BASELINE_TEMPERATURE = 0.2
CHECKPOINT_NAME = "temperature_sensitivity_checkpoint.csv"
MANIFEST_NAME = "manifest.json"
RESULTS_FILENAME = "temperature_sensitivity_results.csv"
SUMMARY_FILENAME = "temperature_sensitivity_summary.csv"
BASELINE_BREAKDOWN = "response_quality_breakdown.csv"

RESULT_COLUMNS = (
    "modelo",
    "temperatura",
    "pais",
    "categoria",
    "dificultad",
    "pregunta",
    "respuesta",
    "judge_label",
)

CHECKPOINT_COLUMNS = ("row_index", *RESULT_COLUMNS, "judge_method", "similarity")

SUMMARY_COLUMNS = (
    "modelo",
    "temperatura",
    "pct_correcta",
    "pct_parcial",
    "pct_alucinacion",
    "pct_abstencion",
    "n_judged",
)


def task_key(row_index: int, model: str, temperature: float) -> tuple[int, str, float]:
    return int(row_index), str(model), round(float(temperature), 1)


def checkpoint_dir_for(output_dir: Path, sample_n: Optional[int]) -> Path:
    if sample_n is None:
        return output_dir / "checkpoints_temperature_sensitivity"
    return output_dir / f"checkpoints_temperature_sensitivity_n{sample_n}"


def load_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(CHECKPOINT_COLUMNS))
    df = pd.read_csv(path)
    for col in CHECKPOINT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[list(CHECKPOINT_COLUMNS)]


def save_checkpoint(path: Path, df: pd.DataFrame, manifest_path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[list(CHECKPOINT_COLUMNS)].to_csv(path, index=False, encoding="utf-8")
    manifest["rows_saved"] = int(len(df))
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def done_keys(checkpoint: pd.DataFrame) -> set[tuple[int, str, float]]:
    if checkpoint.empty:
        return set()
    keys: set[tuple[int, str, float]] = set()
    for row in checkpoint.itertuples(index=False):
        response = getattr(row, "respuesta", "")
        label = getattr(row, "judge_label", "")
        if pd.isna(response) or not str(response).strip():
            continue
        if pd.isna(label) or not str(label).strip():
            continue
        keys.add(task_key(int(row.row_index), row.modelo, row.temperatura))
    return keys


def build_task_list(
    df: pd.DataFrame,
    *,
    models: tuple[str, ...] = MODELS,
    temperatures: tuple[float, ...] = NEW_TEMPERATURES,
) -> list[tuple[int, str, float]]:
    tasks: list[tuple[int, str, float]] = []
    for temp in temperatures:
        for model in models:
            for idx in range(len(df)):
                tasks.append((idx, model, temp))
    return tasks


def generate_response(model: str, question: str, temperature: float) -> Optional[str]:
    if model == "GPT":
        return query_gpt(question, temperature=temperature)
    if model == "Claude":
        return query_claude(question, temperature=temperature)
    raise ValueError(f"Unsupported model: {model}")


def aggregate_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (modelo, temperatura), group in results.groupby(["modelo", "temperatura"], sort=True):
        n = len(group)
        counts = group["judge_label"].value_counts()
        row = {
            "modelo": modelo,
            "temperatura": round(float(temperatura), 1),
            "n_judged": n,
        }
        for label in QUALITY_LABELS:
            count = int(counts.get(label, 0))
            row[f"pct_{label}"] = round(100.0 * count / n, 2) if n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_baseline_summary(results_dir: Path) -> pd.DataFrame:
    path = results_dir / BASELINE_BREAKDOWN
    if not path.exists():
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))

    baseline = pd.read_csv(path)
    rows: list[dict] = []
    for _, row in baseline.iterrows():
        rows.append(
            {
                "modelo": row["model"],
                "temperatura": BASELINE_TEMPERATURE,
                "pct_correcta": row["pct_correcta"],
                "pct_parcial": row["pct_parcial"],
                "pct_alucinacion": row["pct_alucinacion"],
                "pct_abstencion": row["pct_abstencion"],
                "n_judged": int(row["n_judged"]),
            }
        )
    return pd.DataFrame(rows)


def build_full_summary(new_results: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    new_summary = aggregate_summary(new_results)
    baseline_summary = load_baseline_summary(results_dir)
    combined = pd.concat([baseline_summary, new_summary], ignore_index=True)
    combined = combined.sort_values(["modelo", "temperatura"]).reset_index(drop=True)
    return combined[list(SUMMARY_COLUMNS)]


def run_temperature_sensitivity(
    *,
    data_path: Path = LOCAL_SAMPLE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_n: Optional[int] = None,
    models: tuple[str, ...] = MODELS,
    temperatures: tuple[float, ...] = NEW_TEMPERATURES,
    checkpoint_every: int = 10,
    resume: bool = True,
    show_progress: bool = True,
    scorer: Optional[SemanticSimilarityScorer] = None,
) -> dict[str, Path | pd.DataFrame]:
    df = add_group_columns(load_choclo(data_path))
    if sample_n is not None:
        df = df.head(sample_n).reset_index(drop=True)

    checkpoint_dir = checkpoint_dir_for(output_dir, sample_n)
    checkpoint_path = checkpoint_dir / CHECKPOINT_NAME
    manifest_path = checkpoint_dir / MANIFEST_NAME

    manifest = {
        "experiment": "temperature_sensitivity",
        "sample_n": sample_n,
        "n_questions": len(df),
        "models": list(models),
        "temperatures": list(temperatures),
        "baseline_temperature_reused": BASELINE_TEMPERATURE,
        "seed_used": False,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
    }

    checkpoint = load_checkpoint(checkpoint_path) if resume else pd.DataFrame(columns=list(CHECKPOINT_COLUMNS))
    completed = done_keys(checkpoint) if resume else set()
    tasks = build_task_list(df, models=models, temperatures=temperatures)

    total_tasks = len(tasks)
    pending_tasks = total_tasks - len(completed)

    if show_progress:
        log_phase(
            f"Temperature sensitivity — {len(df)} preguntas | "
            f"{len(models)} modelos | temps {list(temperatures)} | "
            f"{total_tasks} tareas ({pending_tasks} pendientes)"
        )
        log_step(
            f"Generacion + judge: ~{total_tasks * 2:,} llamadas API "
            f"(sin seed; baseline T={BASELINE_TEMPERATURE} no regenerado)"
        )
        if resume and completed:
            log_step(f"Reanudando desde checkpoint: {len(completed)}/{total_tasks} tareas hechas")

    scorer = scorer or SemanticSimilarityScorer()
    rows = checkpoint.to_dict(orient="records")
    pending_since_save = 0

    progress = (
        task_progress(total_tasks, "Temp sensitivity", unit="tarea", initial=len(completed))
        if show_progress
        else None
    )

    for row_index, model, temperature in tasks:
        key = task_key(row_index, model, temperature)
        if resume and key in completed:
            if progress is not None:
                progress.update()
            continue

        question_row = df.iloc[row_index]
        question = str(question_row["Question"])
        reference = str(question_row["Answer"])

        response = generate_response(model, question, temperature)
        if not response:
            raise RuntimeError(
                f"Empty generation for row={row_index}, model={model}, temperature={temperature}"
            )

        similarity = float(
            score_embedding([response], [reference], scorer=scorer)[0]
        )
        label, _, judge_method = llm_judge_classify(
            question=question,
            reference=reference,
            response=response,
            score=similarity,
        )

        rows.append(
            {
                "row_index": row_index,
                "modelo": model,
                "temperatura": temperature,
                "pais": question_row["pais"],
                "categoria": question_row["Category"],
                "dificultad": question_row["Difficulty"],
                "pregunta": question,
                "respuesta": response,
                "judge_label": label,
                "judge_method": judge_method,
                "similarity": round(similarity, 4),
            }
        )
        completed.add(key)
        pending_since_save += 1

        if pending_since_save >= checkpoint_every:
            save_checkpoint(
                checkpoint_path,
                pd.DataFrame(rows),
                manifest_path,
                manifest,
            )
            if show_progress:
                log_step(
                    f"Checkpoint guardado ({len(rows)} filas, {checkpoint_dir.name}/"
                    f"{CHECKPOINT_NAME})"
                )
            pending_since_save = 0

        if progress is not None:
            progress.update()

    if progress is not None:
        progress.close()

    results_df = pd.DataFrame(rows)[list(RESULT_COLUMNS)]
    save_checkpoint(checkpoint_path, pd.DataFrame(rows), manifest_path, manifest)

    results_path = output_dir / RESULTS_FILENAME
    if sample_n is not None:
        results_path = output_dir / f"temperature_sensitivity_results_n{sample_n}.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False, encoding="utf-8")

    summary_path = output_dir / SUMMARY_FILENAME
    if sample_n is not None:
        summary_path = output_dir / f"temperature_sensitivity_summary_n{sample_n}.csv"
        summary_df = aggregate_summary(results_df)
    else:
        summary_df = build_full_summary(results_df, output_dir)

    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    if show_progress:
        log_step(f"Resultados: {results_path.name} ({len(results_df):,} filas)")
        log_step(f"Resumen: {summary_path.name} ({len(summary_df):,} filas modelo×temperatura)")

    return {
        "results_df": results_df,
        "summary_df": summary_df,
        "results_path": results_path,
        "summary_path": summary_path,
        "checkpoint_path": checkpoint_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and judge CHOCLO responses at temperature 0.4 and 0.6 "
            "(baseline 0.2 reused from existing evaluation)."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=LOCAL_SAMPLE_PATH,
        help=f"Input CHOCLO CSV (default: {LOCAL_SAMPLE_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Quick test on the first N questions (e.g. 10).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save checkpoint every N completed tasks (default: 10).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoint and start fresh.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env()

    outputs = run_temperature_sensitivity(
        data_path=args.data,
        output_dir=args.output_dir,
        sample_n=args.sample_n,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
        show_progress=not args.quiet,
    )

    summary = outputs["summary_df"]
    print("\nResumen modelo × temperatura:")
    print(summary.to_string(index=False))
    print(f"\nOutputs in {args.output_dir.resolve()}")
    print(f"  - {outputs['results_path'].name}")
    print(f"  - {outputs['summary_path'].name}")


if __name__ == "__main__":
    main()
