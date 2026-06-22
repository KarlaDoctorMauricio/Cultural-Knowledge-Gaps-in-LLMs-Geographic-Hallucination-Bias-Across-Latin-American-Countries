"""End-to-end CHOCLO LLM fairness evaluation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .env import load_env

load_env()

LLM_EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(LLM_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_EVAL_ROOT))

from clients import (  # noqa: E402
    QUERY_FUNCTIONS,
    get_latamgpt_unavailability_note,
    get_query_error,
    reset_latamgpt_status,
)
from .fairness_metrics import regression_fairness_summary, select_best_model_multiobjective
from .postprocessing import apply_postprocessing

from .choclo import (
    CHOCLO_HF_CSV,
    add_group_columns,
    load_choclo,
    normalize_country,
    sample_choclo,
)
from .checkpoint import (
    clear_checkpoints,
    default_checkpoint_dir,
    init_run_manifest,
    load_responses_checkpoint,
    response_is_done,
    save_responses_checkpoint,
)
from .embeddings import SemanticSimilarityScorer
from .progress import count_response_tasks, log_phase, log_step, task_progress
from .group_mae_stats import build_group_mae_category_tables, build_group_mae_tables
from .llm_clients import DEFAULT_MODEL_NAMES, LLMClient

try:
    from evaluator import run_hallucination_analysis
except ImportError:
    run_hallucination_analysis = None

PERFECT_SIMILARITY_TARGET = 1.0
DEFAULT_RUN_MODELS = ("GPT", "Claude")


def _valid_response_mask(responses: pd.Series) -> pd.Series:
    return responses.notna() & responses.astype(str).str.strip().ne("")


def generate_responses(
    df: pd.DataFrame,
    clients: Optional[Dict[str, LLMClient]] = None,
    model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_every: int = 10,
    resume: bool = True,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Generate model answers for each CHOCLO question."""
    result = add_group_columns(df)
    for model_name in model_names:
        result[f"response_{model_name}"] = None
        result[f"note_{model_name}"] = None

    if checkpoint_dir is not None and resume:
        result = load_responses_checkpoint(checkpoint_dir, result, model_names)

    if clients is None:
        reset_latamgpt_status()

    total_tasks, done_tasks = count_response_tasks(
        result,
        model_names,
        resume=resume,
        response_is_done_fn=response_is_done,
    )
    pending_tasks = total_tasks - done_tasks
    if show_progress:
        log_step(
            f"Respuestas LLM: {len(result)} preguntas x {len(model_names)} modelos "
            f"= {total_tasks} llamadas API ({pending_tasks} pendientes)"
        )
        if resume and done_tasks:
            log_step(f"Reanudando desde checkpoint: {done_tasks}/{total_tasks} ya hechas")

    progress = (
        task_progress(total_tasks, "Respuestas", unit="llamada", initial=done_tasks)
        if show_progress
        else None
    )

    pending_since_save = 0

    def _maybe_save_checkpoint(*, force: bool = False) -> None:
        nonlocal pending_since_save
        if checkpoint_dir is None:
            return
        if not force and pending_since_save < checkpoint_every:
            return
        save_responses_checkpoint(
            checkpoint_dir,
            result,
            n_rows=len(result),
            model_names=model_names,
        )
        if show_progress:
            log_step(
                f"Checkpoint respuestas guardado "
                f"({checkpoint_dir.name}/responses_checkpoint.csv)"
            )
        pending_since_save = 0

    for row_idx, row in result.iterrows():
        for model_name in model_names:
            response_col = f"response_{model_name}"
            note_col = f"note_{model_name}"
            if resume and response_is_done(result.at[row_idx, response_col]):
                continue

            if clients is not None:
                if model_name not in clients:
                    raise KeyError(f"Missing client for model: {model_name}")
                client = clients[model_name]
                answer = client.generate(
                    row.Question,
                    country=normalize_country(row.Country),
                    category=row.Category,
                    difficulty=row.Difficulty,
                    reference_answer=row.Answer,
                )
                result.at[row_idx, response_col] = answer
                result.at[row_idx, note_col] = None
            else:
                if model_name not in QUERY_FUNCTIONS:
                    raise KeyError(f"Missing query function for model: {model_name}")

                query_fn = QUERY_FUNCTIONS[model_name]
                answer = query_fn(row.Question)
                if answer is None:
                    result.at[row_idx, response_col] = None
                    if model_name == "LatamGPT":
                        result.at[row_idx, note_col] = (
                            get_latamgpt_unavailability_note()
                            or "LatamGPT no respondio"
                        )
                    else:
                        error_note = get_query_error(model_name)
                        result.at[row_idx, note_col] = (
                            error_note or f"{model_name} no respondio"
                        )
                else:
                    result.at[row_idx, response_col] = answer
                    result.at[row_idx, note_col] = None

            if progress is not None:
                progress.set_postfix_str(f"{model_name} fila {row_idx}", refresh=False)
                progress.update(1)

            pending_since_save += 1
            _maybe_save_checkpoint()

    if progress is not None:
        progress.close()

    if checkpoint_dir is not None:
        _maybe_save_checkpoint(force=True)

    return result


def score_responses(
    df: pd.DataFrame,
    model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
    scorer: Optional[SemanticSimilarityScorer] = None,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Compute semantic similarity scores and MAE-compatible predictions.

    Missing responses are stored as ``NaN`` similarity/error values.
    """
    scorer = scorer or SemanticSimilarityScorer()
    result = add_group_columns(df) if "pais" not in df.columns else df.copy()
    preds_dict: Dict[str, np.ndarray] = {}

    references = result["Answer"].astype(str).tolist()

    if show_progress:
        log_step(f"Embeddings: {len(model_names)} modelos, {len(result)} preguntas")

    for model_idx, model_name in enumerate(model_names, start=1):
        response_col = f"response_{model_name}"
        if response_col not in result.columns:
            raise KeyError(f"Missing response column: {response_col}")

        responses = result[response_col]
        valid_mask = _valid_response_mask(responses)
        similarities = np.full(len(result), np.nan, dtype=float)

        if show_progress:
            log_step(
                f"  [{model_idx}/{len(model_names)}] {model_name}: "
                f"{int(valid_mask.sum())} respuestas validas"
            )

        if valid_mask.any():
            valid_responses = responses[valid_mask].astype(str).tolist()
            valid_refs = [references[i] for i, ok in enumerate(valid_mask.to_numpy()) if ok]
            valid_scores = scorer.score(valid_responses, valid_refs)
            similarities[valid_mask.to_numpy()] = valid_scores
            preds_dict[model_name] = similarities

        result[f"similarity_{model_name}"] = similarities
        result[f"error_{model_name}"] = PERFECT_SIMILARITY_TARGET - similarities

    result["y_true"] = PERFECT_SIMILARITY_TARGET
    return result, preds_dict


def evaluate_available_models(
    preds_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    sensitive: np.ndarray,
    config_name: str,
    df_group_pais: Optional[pd.DataFrame] = None,
    min_n: int = 5,
) -> list[dict]:
    """Evaluate only rows with valid predictions; unavailable models get NaN metrics."""
    results = []

    for method, preds in preds_dict.items():
        valid = ~np.isnan(preds)
        n_valid = int(valid.sum())

        if n_valid == 0:
            results.append(
                {
                    "config": config_name,
                    "method": method,
                    "mae": np.nan,
                    "bias": np.nan,
                    "tradeoff": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "effect_size": np.nan,
                    "n_valid": 0,
                    "note": "Modelo no disponible",
                    "nota_robustez": None,
                }
            )
            continue

        summary = regression_fairness_summary(
            y_true[valid],
            preds[valid],
            sensitive[valid],
        )
        row = {
            "config": config_name,
            "method": method,
            "mae": summary["mae_global"],
            "bias": summary["bias"],
            "tradeoff": summary["tradeoff_score"],
            "ci_lower": summary["ci_lower"],
            "ci_upper": summary["ci_upper"],
            "effect_size": summary["effect_size"],
            "n_valid": n_valid,
            "note": None,
            "nota_robustez": None,
        }
        if df_group_pais is not None and not df_group_pais.empty:
            from .group_mae_stats import assess_bias_robustness

            row["nota_robustez"] = assess_bias_robustness(
                df_group_pais, method, min_n=min_n
            )
        results.append(row)

    return results


def run_choclo_evaluation(
    df: pd.DataFrame,
    config_name: str = "CHOCLO",
    clients: Optional[Dict[str, LLMClient]] = None,
    scorer: Optional[SemanticSimilarityScorer] = None,
    model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
    lambda_fairness: float = 0.5,
    apply_calibration: bool = True,
    calibration_targets: Sequence[str] | str = "all",
    min_group_n: int = 5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_every: int = 10,
    resume: bool = True,
    clear_checkpoint_on_success: bool = True,
    show_progress: bool = False,
) -> Dict[str, pd.DataFrame | Dict[str, np.ndarray] | str | list[str]]:
    """
    Run the full CHOCLO evaluation and feed results into existing fairness utilities.

    If LatamGPT is unavailable, the pipeline continues with the remaining models.
    Partial progress is saved under ``checkpoint_dir`` every ``checkpoint_every`` tasks.
    """
    judge_models = [m for m in ("GPT", "Claude") if m in model_names]
    if checkpoint_dir is not None:
        init_run_manifest(
            checkpoint_dir,
            n_rows=len(df),
            model_names=model_names,
            judge_models=judge_models,
        )

    if show_progress:
        log_phase(
            f"Evaluacion — {len(df)} preguntas | modelos: {', '.join(model_names)}"
        )
        log_step("Paso 1/3: generar respuestas LLM")

    responses_df = generate_responses(
        df,
        clients=clients,
        model_names=model_names,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        resume=resume,
        show_progress=show_progress,
    )

    if show_progress:
        log_step("Paso 2/3: embeddings y similitud semantica")

    scored_df, preds_dict = score_responses(
        responses_df,
        model_names=model_names,
        scorer=scorer,
        show_progress=show_progress,
    )

    unavailable_models = [
        model
        for model in model_names
        if model not in preds_dict or np.isnan(preds_dict[model]).all()
    ]
    available_models = [model for model in model_names if model in preds_dict]

    if not available_models:
        raise RuntimeError("Ningun modelo devolvio respuestas validas.")

    y_true = scored_df["y_true"].to_numpy(dtype=float)
    pais = scored_df["pais"].to_numpy()
    interseccion = scored_df["interseccion"].to_numpy()
    category = scored_df["Category"].astype(str).to_numpy()

    df_group_mae_pais, df_group_mae_interseccion = build_group_mae_tables(
        y_true,
        preds_dict,
        pais,
        interseccion,
        config_name,
        min_n=min_group_n,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    df_group_mae_category = build_group_mae_category_tables(
        y_true,
        preds_dict,
        category,
        config_name,
        min_n=min_group_n,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    df_metrics = pd.DataFrame(
        evaluate_available_models(
            preds_dict,
            y_true,
            pais,
            config_name,
            df_group_pais=df_group_mae_pais,
            min_n=min_group_n,
        )
    )

    df_metrics_valid = df_metrics.dropna(subset=["mae"])
    if df_metrics_valid.empty:
        raise RuntimeError("No hay metricas validas para seleccionar un modelo.")

    df_selection = select_best_model_multiobjective(
        df_metrics_valid,
        df_group_mae_interseccion.rename(columns={"interseccion": "group"}),
        lambda_fairness=lambda_fairness,
    )
    best_method = str(df_selection.iloc[0]["method"])

    preds_post: Dict[str, np.ndarray] = {}
    df_metrics_post = pd.DataFrame()
    df_group_mae_pais_post = pd.DataFrame()
    df_group_mae_interseccion_post = pd.DataFrame()
    df_group_mae_category_post = pd.DataFrame()

    if apply_calibration:
        preds_post = apply_postprocessing(
            preds_dict,
            pais,
            target_models=calibration_targets,
        )
        df_group_mae_pais_post, df_group_mae_interseccion_post = build_group_mae_tables(
            y_true,
            preds_post,
            pais,
            interseccion,
            config_name,
            min_n=min_group_n,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        df_group_mae_category_post = build_group_mae_category_tables(
            y_true,
            preds_post,
            category,
            config_name,
            min_n=min_group_n,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        df_metrics_post = pd.DataFrame(
            evaluate_available_models(
                preds_post,
                y_true,
                pais,
                config_name,
                df_group_pais=df_group_mae_pais_post,
                min_n=min_group_n,
            )
        )

    latamgpt_note = get_latamgpt_unavailability_note()
    if latamgpt_note and "LatamGPT" in unavailable_models:
        scored_df.loc[scored_df["note_LatamGPT"].isna(), "note_LatamGPT"] = latamgpt_note

    judge_models = [m for m in ("GPT", "Claude") if m in available_models]
    low_score_classifications = pd.DataFrame()
    response_quality_breakdown = pd.DataFrame()
    response_quality_by_country = pd.DataFrame()
    if run_hallucination_analysis is not None and judge_models:
        if show_progress:
            log_step(
                f"Paso 3/3: judge LLM ({', '.join(judge_models)}) — "
                "clasificar correcta/parcial/alucinacion/abstencion"
            )
        low_score_classifications, response_quality_breakdown, response_quality_by_country = (
            run_hallucination_analysis(
                scored_df,
                model_names=judge_models,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=checkpoint_every,
                resume=resume,
                show_progress=show_progress,
            )
        )
        if show_progress and not low_score_classifications.empty:
            log_step(
                f"Judge completado: {len(low_score_classifications):,} clasificaciones"
            )

    if checkpoint_dir is not None and clear_checkpoint_on_success:
        clear_checkpoints(checkpoint_dir)

    return {
        "scored_df": scored_df,
        "preds_dict": preds_dict,
        "preds_post": preds_post,
        "df_metrics": df_metrics,
        "df_metrics_post": df_metrics_post,
        "df_group_mae_pais": df_group_mae_pais,
        "df_group_mae_interseccion": df_group_mae_interseccion,
        "df_group_mae_category": df_group_mae_category,
        "df_group_mae_pais_post": df_group_mae_pais_post,
        "df_group_mae_interseccion_post": df_group_mae_interseccion_post,
        "df_group_mae_category_post": df_group_mae_category_post,
        "df_selection": df_selection,
        "best_method": best_method,
        "config_name": config_name,
        "available_models": available_models,
        "unavailable_models": unavailable_models,
        "latamgpt_note": latamgpt_note,
        "low_score_classifications": low_score_classifications,
        "response_quality_breakdown": response_quality_breakdown,
        "response_quality_by_country": response_quality_by_country,
        # backward-compatible keys
        "hallucination_vs_abstention": response_quality_breakdown,
        "hallucination_by_country": response_quality_by_country,
    }


def run_from_sample(
    n: int = 30,
    random_state: int = 42,
    data_path: Optional[str | Path] = None,
    **kwargs,
) -> Dict[str, pd.DataFrame | Dict[str, np.ndarray] | str | list[str]]:
    """
    Convenience wrapper: load CHOCLO and evaluate a random subset.

    Uses the official Hugging Face dataset by default (``CHOCLO_HF_CSV``).
    """
    df = sample_choclo(load_choclo(data_path), n=n, random_state=random_state)
    return run_choclo_evaluation(df, **kwargs)
