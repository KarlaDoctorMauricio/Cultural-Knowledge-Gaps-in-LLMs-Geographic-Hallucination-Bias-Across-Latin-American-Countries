"""
CHOCLO response evaluation: embedding similarity and LLM-judge quality typing.

An LLM judge classifies each evaluated response into one of four CHOCLO-aligned
categories: ``correcta``, ``parcial``, ``alucinacion``, or ``abstencion``.
GPT is the primary judge; Claude is the backup; regex heuristics are last resort.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

from clients import query_claude, query_gpt  # noqa: E402

from fairness_toolkit.checkpoint import (  # noqa: E402
    judge_done_keys,
    judge_task_key,
    load_judge_checkpoint,
    save_judge_checkpoint,
)
from fairness_toolkit.progress import log_step, task_progress  # noqa: E402

EMBEDDINGS_MODULE_PATH = ROOT / "fairness_toolkit" / "embeddings.py"


def _load_embeddings_module():
    spec = importlib.util.spec_from_file_location("embeddings_module", EMBEDDINGS_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_embeddings = _load_embeddings_module()
SemanticSimilarityScorer = _embeddings.SemanticSimilarityScorer

LOW_EMBEDDING_THRESHOLD = 0.3  # kept for optional reporting filters only
JUDGE_MODELS = ("GPT", "Claude")
QUALITY_LABELS = ("correcta", "parcial", "alucinacion", "abstencion")
JUDGE_LABELS = QUALITY_LABELS  # backward-compatible alias
JUDGE_METHODS = ("gpt", "claude", "heuristic_fallback")

DEFAULT_RESULTS = ROOT / "data" / "results" / "evaluation_results.csv"
DEFAULT_SUMMARY_OUTPUT = ROOT / "data" / "results" / "response_quality_breakdown.csv"
DEFAULT_COUNTRY_OUTPUT = ROOT / "data" / "results" / "hallucination_by_country.csv"
DEFAULT_DETAIL_OUTPUT = ROOT / "data" / "results" / "low_score_classifications.csv"

JUDGE_PROMPT_VERSION = "v2_fixed_system_prompt"

JUDGE_SYSTEM_PROMPT = """Eres un evaluador de respuestas de modelos de lenguaje sobre conocimiento cultural latinoamericano.

Clasifica cada respuesta del modelo en exactamente UNA de cuatro categorias alineadas con el criterio del benchmark CHOCLO:

- CORRECTA: coincide en sustancia con el Answer esperado (el dato esencial es correcto, aunque la redaccion varie levemente).
- PARCIAL: captura el dominio, tipo de entidad o categoria correcta, pero falla en precision o especificidad del dato exacto.
- ALUCINACION: afirma hechos concretos (nombres, lugares, fechas, datos especificos) incorrectos o sin relacion real con el Answer esperado.
- ABSTENCION: indica explicitamente que no tiene la informacion, no puede responder con certeza, o pide mas contexto SIN inventar un hecho concreto como respuesta final.

Ejemplos orientativos:
- CORRECTA: Answer "Contrabajo" y respuesta "Contrabajo" (coincide en sustancia).
- PARCIAL: pregunta sobre Cerros de la Calera (Uruguay); Answer "localidad de Rivera" y respuesta "area natural protegida" (entiende el tipo de entidad geografica pero no el dato exacto).
- ALUCINACION: Answer "Contrabajo" y respuesta "Oscar Alem toco el acordeon antes del piano"; Answer "Corrientes" y respuesta "Antonio Gonzaga nacio en Italia" (respuesta concreta pero incorrecta, sin relacion real con la referencia).
- ABSTENCION: "No tengo informacion especifica sobre Oscar Alem..." o pide mas contexto sin afirmar un dato concreto como respuesta final.

Responde siempre con una sola palabra: CORRECTA, PARCIAL, ALUCINACION o ABSTENCION."""

JUDGE_USER_TEMPLATE = """Pregunta:
{question}

Respuesta de referencia (Answer esperado):
{reference}

Respuesta del modelo:
{response}

Similitud semantica con la referencia (embedding): {score:.3f}."""

ABSTENTION_PATTERNS = (
    r"no tengo informaci[oó]n",
    r"no dispongo de informaci[oó]n",
    r"no cuento con informaci[oó]n",
    r"no tengo datos",
    r"no s[eé]\b",
    r"no puedo (?:responder|confirmar|determinar|identificar)",
    r"no (?:estoy|soy) seguro",
    r"sin m[aá]s contexto",
    r"m[aá]s contexto",
    r"m[aá]s detalles",
    r"podr[ií]as proporcionar",
    r"no (?:encuentro|localizo) informaci[oó]n",
    r"no (?:tengo|hay) (?:acceso|registro)",
    r"informaci[oó]n limitada",
    r"no (?:puedo|soy capaz de) (?:verificar|validar)",
)


def score_embedding(
    predictions: Sequence[str],
    references: Sequence[str],
    scorer: Optional[SemanticSimilarityScorer] = None,
) -> np.ndarray:
    """Compute cosine embedding similarity for prediction/reference pairs."""
    scorer = scorer or SemanticSimilarityScorer()
    return scorer.score(predictions, references)


def _embedding_column(model: str) -> str:
    return f"score_embedding_{model}"


def _similarity_column(model: str) -> str:
    return f"similarity_{model}"


def _response_column(model: str) -> str:
    return f"response_{model}"


def ensure_embedding_score_columns(
    df: pd.DataFrame,
    model_names: Sequence[str] = JUDGE_MODELS,
) -> pd.DataFrame:
    """Ensure ``score_embedding_{model}`` columns exist (alias of similarity scores)."""
    result = df.copy()
    for model in model_names:
        embed_col = _embedding_column(model)
        sim_col = _similarity_column(model)
        if embed_col in result.columns:
            continue
        if sim_col in result.columns:
            result[embed_col] = result[sim_col]
        else:
            result[embed_col] = np.nan
    return result


def _normalize_judge_label(text: str) -> Optional[str]:
    cleaned = text.strip().lower()
    cleaned = cleaned.replace("á", "a").replace("é", "e").replace("í", "i")
    cleaned = cleaned.replace("ó", "o").replace("ú", "u")

    if "abstencion" in cleaned or "abstention" in cleaned:
        return "abstencion"
    if "alucinacion" in cleaned or "hallucin" in cleaned:
        return "alucinacion"
    if "parcial" in cleaned or "partial" in cleaned:
        return "parcial"
    if "correcta" in cleaned or "correct" in cleaned:
        return "correcta"
    return None


def rule_based_judge_classify(response: str, score: Optional[float] = None) -> str:
    """Heuristic fallback when both LLM judges are unavailable."""
    text = str(response).strip().lower()
    if not text:
        return "abstencion"

    for pattern in ABSTENTION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return "abstencion"

    if score is not None:
        if score >= 0.75:
            return "correcta"
        if score >= 0.45:
            return "parcial"

    return "alucinacion"


def _call_judge(
    query_fn: Callable[..., Optional[str]],
    user_prompt: str,
    *,
    system_prompt: str = JUDGE_SYSTEM_PROMPT,
) -> tuple[Optional[str], Optional[str]]:
    """Return ``(label, raw_text)`` when the judge response is parseable."""
    raw = query_fn(user_prompt, system_prompt=system_prompt)
    if not raw:
        return None, None

    raw = raw.strip()
    label = _normalize_judge_label(raw)
    if label is None:
        return None, raw
    return label, raw


def llm_judge_classify(
    question: str,
    reference: str,
    response: str,
    score: float,
    primary_query_fn: Callable[..., Optional[str]] = query_gpt,
    backup_query_fn: Callable[..., Optional[str]] = query_claude,
    *,
    system_prompt: str = JUDGE_SYSTEM_PROMPT,
) -> tuple[str, Optional[str], str]:
    """
    Classify a response into ``correcta``, ``parcial``, ``alucinacion``, or ``abstencion``.

    Returns ``(label, raw_judge_text, judge_method)``. Tries GPT first, then
    Claude; only if both fail uses ``heuristic_fallback``.
    """
    user_prompt = JUDGE_USER_TEMPLATE.format(
        question=question.strip(),
        reference=reference.strip(),
        response=response.strip(),
        score=float(score),
    )

    label, raw = _call_judge(
        primary_query_fn,
        user_prompt,
        system_prompt=system_prompt,
    )
    if label is not None:
        return label, raw, "gpt"

    label, raw_backup = _call_judge(
        backup_query_fn,
        user_prompt,
        system_prompt=system_prompt,
    )
    if label is not None:
        return label, raw_backup, "claude"

    fallback = rule_based_judge_classify(response, score=score)
    last_raw = raw_backup or raw
    return fallback, last_raw, "heuristic_fallback"


def _count_judge_tasks(
    df: pd.DataFrame,
    model_names: Sequence[str],
    *,
    done_keys: set[tuple[int, str]],
    resume: bool,
) -> tuple[int, int]:
    """Return (total_judge_tasks, already_done)."""
    working = ensure_embedding_score_columns(df, model_names=model_names)
    total = 0
    for model in model_names:
        embed_col = _embedding_column(model)
        response_col = _response_column(model)
        if embed_col not in working.columns or response_col not in working.columns:
            continue

        subset = working[working[embed_col].notna()]
        for idx, row in subset.iterrows():
            response = row[response_col]
            if pd.isna(response) or not str(response).strip():
                continue
            total += 1

    done = len(done_keys) if resume else 0
    return total, done


def classify_low_score_responses(
    df: pd.DataFrame,
    model_names: Sequence[str] = JUDGE_MODELS,
    primary_query_fn: Callable[[str], Optional[str]] = query_gpt,
    backup_query_fn: Callable[[str], Optional[str]] = query_claude,
    use_llm_judge: bool = True,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_every: int = 10,
    resume: bool = True,
    show_progress: bool = False,
    judge_prompt_version: Optional[str] = JUDGE_PROMPT_VERSION,
) -> pd.DataFrame:
    """Classify every valid evaluated response per model."""
    working = ensure_embedding_score_columns(df, model_names=model_names)
    rows: list[dict] = []
    done_keys: set[tuple[int, str]] = set()

    if checkpoint_dir is not None and resume:
        existing = load_judge_checkpoint(checkpoint_dir)
        if not existing.empty:
            rows.extend(existing.to_dict(orient="records"))
            done_keys = judge_done_keys(existing)

    total_tasks, done_tasks = _count_judge_tasks(
        working,
        model_names,
        done_keys=done_keys,
        resume=resume,
    )
    pending_tasks = total_tasks - done_tasks
    if show_progress:
        log_step(
            f"Judge: {total_tasks} respuestas a clasificar "
            f"({pending_tasks} pendientes, modelos: {', '.join(model_names)})"
        )
        if resume and done_tasks:
            log_step(f"Reanudando judge desde checkpoint: {done_tasks}/{total_tasks}")

    progress = (
        task_progress(total_tasks, "Judge", unit="resp", initial=done_tasks)
        if show_progress
        else None
    )

    pending_since_save = 0

    def _flush_checkpoint(*, force: bool = False) -> None:
        nonlocal pending_since_save
        if checkpoint_dir is None:
            return
        if not force and pending_since_save < checkpoint_every:
            return
        save_judge_checkpoint(checkpoint_dir, pd.DataFrame(rows))
        if show_progress:
            log_step(
                f"Checkpoint judge guardado "
                f"({len(rows)} clasificaciones, {checkpoint_dir.name}/judge_checkpoint.csv)"
            )
        pending_since_save = 0

    for model in model_names:
        embed_col = _embedding_column(model)
        response_col = _response_column(model)
        if embed_col not in working.columns or response_col not in working.columns:
            continue

        subset = working[working[embed_col].notna()].copy()

        for idx, row in subset.iterrows():
            response = row[response_col]
            if pd.isna(response) or not str(response).strip():
                continue

            task_key = judge_task_key(idx, model)
            if resume and task_key in done_keys:
                continue

            score = float(row[embed_col])
            if use_llm_judge:
                label, judge_raw, judge_method = llm_judge_classify(
                    question=str(row["Question"]),
                    reference=str(row["Answer"]),
                    response=str(response),
                    score=score,
                    primary_query_fn=primary_query_fn,
                    backup_query_fn=backup_query_fn,
                )
            else:
                label = rule_based_judge_classify(str(response), score=score)
                judge_raw = None
                judge_method = "heuristic_fallback"

            rows.append(
                {
                    "row_index": idx,
                    "model": model,
                    "pais": row.get("pais"),
                    "Category": row.get("Category"),
                    "Difficulty": row.get("Difficulty"),
                    "Question": row["Question"],
                    "Answer": row["Answer"],
                    "response": response,
                    "score_embedding": score,
                    "judge_label": label,
                    "judge_method": judge_method,
                    "judge_raw": judge_raw,
                    "judge_prompt_version": judge_prompt_version,
                }
            )
            done_keys.add(task_key)
            pending_since_save += 1

            if progress is not None:
                progress.set_postfix_str(f"{model} fila {idx}", refresh=False)
                progress.update(1)

            _flush_checkpoint()

    if progress is not None:
        progress.close()

    if checkpoint_dir is not None and rows:
        _flush_checkpoint(force=True)

    if not rows:
        return pd.DataFrame(
            columns=[
                "row_index",
                "model",
                "pais",
                "Category",
                "Difficulty",
                "Question",
                "Answer",
                "response",
                "score_embedding",
                "judge_label",
                "judge_method",
                "judge_raw",
                "judge_prompt_version",
            ]
        )

    return pd.DataFrame(rows)


def _quality_summary_columns() -> list[str]:
    columns = ["n_judged"]
    for label in QUALITY_LABELS:
        columns.extend([f"n_{label}", f"pct_{label}"])
    return columns


def _aggregate_quality_row(
    subset: pd.DataFrame,
    extra_fields: Optional[dict] = None,
) -> dict:
    """Build one summary row with counts and percentages for all quality labels."""
    row = dict(extra_fields or {})
    n_judged = int(len(subset))
    row["n_judged"] = n_judged

    if n_judged == 0:
        for label in QUALITY_LABELS:
            row[f"n_{label}"] = 0
            row[f"pct_{label}"] = np.nan
        return row

    for label in QUALITY_LABELS:
        count = int((subset["judge_label"] == label).sum())
        row[f"n_{label}"] = count
        row[f"pct_{label}"] = round(100.0 * count / n_judged, 2)
    return row


def build_response_quality_breakdown_table(
    classified: pd.DataFrame,
    model_names: Sequence[str] = JUDGE_MODELS,
) -> pd.DataFrame:
    """Aggregate quality labels into per-model counts and percentages."""
    rows: list[dict] = []

    for model in model_names:
        subset = classified[classified["model"] == model]
        rows.append(_aggregate_quality_row(subset, {"model": model}))

    return pd.DataFrame(rows)


def build_response_quality_by_country_table(
    classified: pd.DataFrame,
    model_names: Sequence[str] = JUDGE_MODELS,
) -> pd.DataFrame:
    """Aggregate quality labels by country and model."""
    empty_columns = ["pais", "model", *_quality_summary_columns()]
    if classified.empty or "pais" not in classified.columns:
        return pd.DataFrame(columns=empty_columns)

    filtered = classified[classified["model"].isin(model_names)].copy()
    if filtered.empty:
        return pd.DataFrame(columns=empty_columns)

    rows: list[dict] = []
    for (pais, model), subset in filtered.groupby(["pais", "model"], sort=True):
        rows.append(
            _aggregate_quality_row(
                subset,
                {"pais": pais, "model": model},
            )
        )

    return pd.DataFrame(rows).sort_values(
        ["model", "pais"],
        ascending=[True, True],
        kind="stable",
    )


# Backward-compatible aliases
build_hallucination_vs_abstention_table = build_response_quality_breakdown_table
build_hallucination_by_country_table = build_response_quality_by_country_table
_aggregate_hallucination_row = _aggregate_quality_row


def run_hallucination_analysis(
    df: pd.DataFrame,
    model_names: Sequence[str] = JUDGE_MODELS,
    primary_query_fn: Callable[[str], Optional[str]] = query_gpt,
    backup_query_fn: Callable[[str], Optional[str]] = query_claude,
    use_llm_judge: bool = True,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_every: int = 10,
    resume: bool = True,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return row-level classifications, global summary, and country breakdown."""
    classified = classify_low_score_responses(
        df,
        model_names=model_names,
        primary_query_fn=primary_query_fn,
        backup_query_fn=backup_query_fn,
        use_llm_judge=use_llm_judge,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        resume=resume,
        show_progress=show_progress,
    )
    summary = build_response_quality_breakdown_table(classified, model_names=model_names)
    by_country = build_response_quality_by_country_table(classified, model_names=model_names)
    return classified, summary, by_country


def reaggregate_hallucination_tables(
    classified: pd.DataFrame,
    model_names: Sequence[str] = JUDGE_MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild summary tables from existing row-level classifications."""
    summary = build_response_quality_breakdown_table(classified, model_names=model_names)
    by_country = build_response_quality_by_country_table(classified, model_names=model_names)
    return summary, by_country


reaggregate_response_quality_tables = reaggregate_hallucination_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify evaluated CHOCLO responses into correcta, parcial, "
            "alucinacion, or abstencion."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Evaluation results CSV (default: {DEFAULT_RESULTS}).",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help=f"Summary CSV path (default: {DEFAULT_SUMMARY_OUTPUT}).",
    )
    parser.add_argument(
        "--detail-output",
        type=Path,
        default=DEFAULT_DETAIL_OUTPUT,
        help=f"Row-level classifications CSV (default: {DEFAULT_DETAIL_OUTPUT}).",
    )
    parser.add_argument(
        "--country-output",
        type=Path,
        default=DEFAULT_COUNTRY_OUTPUT,
        help=f"Country breakdown CSV path (default: {DEFAULT_COUNTRY_OUTPUT}).",
    )
    parser.add_argument(
        "--from-classifications",
        type=Path,
        default=None,
        help=(
            "Skip judge calls and rebuild summary tables from an existing "
            "low_score_classifications.csv."
        ),
    )
    parser.add_argument(
        "--rule-based",
        action="store_true",
        help="Use heuristic abstention detection instead of LLM judge API calls.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save judge progress every N classifications (default: 10).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_RESULTS.parent / "checkpoints",
        help="Checkpoint directory for judge resume.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing judge checkpoint and start fresh.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    if args.from_classifications is not None:
        print(f"Loading classifications from {args.from_classifications} ...")
        classified = pd.read_csv(args.from_classifications)
        summary, by_country = reaggregate_hallucination_tables(classified)
        print(f"Loaded {len(classified):,} classified rows.")
    else:
        print(f"Loading {args.input} ...")
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df):,} rows.")

        classified, summary, by_country = run_hallucination_analysis(
            df,
            use_llm_judge=not args.rule_based,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
            resume=not args.no_resume,
        )
        classified.to_csv(args.detail_output, index=False, encoding="utf-8")
        print(f"Classified {len(classified):,} evaluated responses.")
        print(f"Saved detail to {args.detail_output}")

    summary.to_csv(args.summary_output, index=False, encoding="utf-8")
    by_country.to_csv(args.country_output, index=False, encoding="utf-8")

    print(f"Saved summary to {args.summary_output}")
    print(f"Saved country breakdown to {args.country_output}")
    print("\nSummary by model:")
    print(summary.to_string(index=False))
    print("\nSummary by country (first 20 rows):")
    print(by_country.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
