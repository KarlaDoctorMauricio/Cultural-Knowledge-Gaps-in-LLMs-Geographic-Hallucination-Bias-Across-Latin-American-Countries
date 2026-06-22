"""Group-level MAE statistics with sample-size checks and bootstrap uncertainty."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

INSUFFICIENT_SAMPLE_NOTE = (
    "muestra insuficiente — no interpretar como tendencia"
)
BIAS_NOT_ROBUST_NOTE = (
    "posiblemente no robusto: bias influenciado por grupo(s) con muestra insuficiente"
)


def _bootstrap_group_mae(
    abs_errors: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Return (ci_lower, ci_upper, margin) for a group's MAE."""
    n = len(abs_errors)
    if n == 0:
        return np.nan, np.nan, np.nan
    if n == 1:
        value = float(abs_errors[0])
        return value, value, 0.0

    rng = np.random.default_rng(random_state)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(abs_errors, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    margin = (ci_upper - ci_lower) / 2.0
    return ci_lower, ci_upper, margin


def _bootstrap_proportion(
    labels: np.ndarray,
    target_label: str,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Return (ci_lower, ci_upper, margin) for a label proportion in percent."""
    n = len(labels)
    if n == 0:
        return np.nan, np.nan, np.nan
    if n == 1:
        value = 100.0 if str(labels[0]) == target_label else 0.0
        return value, value, 0.0

    is_target = np.array([str(label) == target_label for label in labels], dtype=float)
    rng = np.random.default_rng(random_state)
    boot_props = []
    for _ in range(n_bootstrap):
        sample = rng.choice(is_target, size=n, replace=True)
        boot_props.append(100.0 * float(np.mean(sample)))

    ci_lower = float(np.percentile(boot_props, 2.5))
    ci_upper = float(np.percentile(boot_props, 97.5))
    margin = (ci_upper - ci_lower) / 2.0
    return ci_lower, ci_upper, margin


def compute_group_mae_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    *,
    group_column: str,
    method: str,
    config_name: str,
    group_level: str,
    min_n: int = 5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build an enriched group MAE table for one model and grouping column."""
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "group": groups,
        }
    )
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])

    rows = []
    for group_value, group_df in df.groupby("group", sort=True):
        abs_errors = group_df["abs_error"].to_numpy(dtype=float)
        n_preguntas = int(len(abs_errors))
        mae = float(np.mean(abs_errors)) if n_preguntas else np.nan
        confiable = n_preguntas >= min_n

        ci_lower, ci_upper, margin = _bootstrap_group_mae(
            abs_errors,
            n_bootstrap=n_bootstrap,
            random_state=random_state + hash(str(group_value)) % 10_000,
        )

        rows.append(
            {
                group_column: group_value,
                "mae": mae,
                "n_preguntas": n_preguntas,
                "confiable": confiable,
                "nota": None if confiable else INSUFFICIENT_SAMPLE_NOTE,
                "mae_ci_lower": ci_lower,
                "mae_ci_upper": ci_upper,
                "mae_margin": margin,
                "method": method,
                "config": config_name,
                "group_level": group_level,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    table = table.sort_values(
        by=["confiable", "mae"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)
    return table


def build_group_mae_tables(
    y_true: np.ndarray,
    preds_dict: dict[str, np.ndarray],
    pais: np.ndarray,
    interseccion: np.ndarray,
    config_name: str,
    min_n: int = 5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build country-only and intersectional MAE tables with uncertainty."""
    pais_tables = []
    interseccion_tables = []

    for method, preds in preds_dict.items():
        valid = ~np.isnan(preds)
        if valid.sum() == 0:
            continue

        y_valid = y_true[valid]
        preds_valid = preds[valid]
        pais_valid = pais[valid]
        interseccion_valid = interseccion[valid]

        pais_tables.append(
            compute_group_mae_table(
                y_valid,
                preds_valid,
                pais_valid,
                group_column="pais",
                method=method,
                config_name=config_name,
                group_level="pais",
                min_n=min_n,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
        )
        interseccion_tables.append(
            compute_group_mae_table(
                y_valid,
                preds_valid,
                interseccion_valid,
                group_column="interseccion",
                method=method,
                config_name=config_name,
                group_level="interseccion",
                min_n=min_n,
                n_bootstrap=n_bootstrap,
                random_state=random_state + 1,
            )
        )

    if not pais_tables:
        return pd.DataFrame(), pd.DataFrame()

    return pd.concat(pais_tables, ignore_index=True), pd.concat(
        interseccion_tables, ignore_index=True
    )


def build_group_mae_category_tables(
    y_true: np.ndarray,
    preds_dict: dict[str, np.ndarray],
    category: np.ndarray,
    config_name: str,
    min_n: int = 5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build category-only MAE tables (no country) with uncertainty."""
    category_tables = []

    for method, preds in preds_dict.items():
        valid = ~np.isnan(preds)
        if valid.sum() == 0:
            continue

        category_tables.append(
            compute_group_mae_table(
                y_true[valid],
                preds[valid],
                category[valid],
                group_column="category",
                method=method,
                config_name=config_name,
                group_level="category",
                min_n=min_n,
                n_bootstrap=n_bootstrap,
                random_state=random_state + 2,
            )
        )

    if not category_tables:
        return pd.DataFrame()

    return pd.concat(category_tables, ignore_index=True)


def assess_bias_robustness(
    df_group_pais: pd.DataFrame,
    method: str,
    min_n: int = 5,
) -> Optional[str]:
    """
    Flag bias as potentially unreliable when extreme MAE groups have small n.
    """
    subset = df_group_pais[df_group_pais["method"] == method]
    if subset.empty or len(subset) < 2:
        return None

    max_row = subset.loc[subset["mae"].idxmax()]
    min_row = subset.loc[subset["mae"].idxmin()]

    if int(max_row["n_preguntas"]) < min_n or int(min_row["n_preguntas"]) < min_n:
        return BIAS_NOT_ROBUST_NOTE

    return None


def add_robustness_notes(
    df_metrics: pd.DataFrame,
    df_group_pais: pd.DataFrame,
    min_n: int = 5,
) -> pd.DataFrame:
    """Add ``nota_robustez`` to metrics based on country-level group sample sizes."""
    result = df_metrics.copy()
    result["nota_robustez"] = result["method"].map(
        lambda method: assess_bias_robustness(df_group_pais, method, min_n=min_n)
    )
    return result
