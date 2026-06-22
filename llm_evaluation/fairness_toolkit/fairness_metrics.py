"""Regression fairness metrics reused from the CHOCLO evaluation pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_fairness_summary(y_true, y_pred, sensitive_features):
    mae_global = np.mean(np.abs(y_true - y_pred))
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": sensitive_features})
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])
    mae_by_group = df.groupby("group")["abs_error"].mean()
    bias = mae_by_group.max() - mae_by_group.min()
    tradeoff_score = mae_global + bias
    ci_lower = mae_global - 0.05 * mae_global
    ci_upper = mae_global + 0.05 * mae_global
    effect_size = bias / (mae_global + 1e-8)
    return {
        "mae_global": mae_global,
        "bias": bias,
        "tradeoff_score": tradeoff_score,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "effect_size": effect_size,
    }


def compute_intersectional_mae(y_true, y_pred, sensitive_features):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": sensitive_features})
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])
    return (
        df.groupby("group")["abs_error"]
        .mean()
        .reset_index()
        .rename(columns={"abs_error": "mae"})
        .sort_values("mae", ascending=False)
    )


def select_best_model_multiobjective(df_metrics, df_group_mae, lambda_fairness=0.5):
    gap_df = df_group_mae.groupby("method")["mae"].agg(["max", "min"]).reset_index()
    gap_df["fairness_gap"] = gap_df["max"] - gap_df["min"]
    df = df_metrics.merge(gap_df[["method", "fairness_gap"]], on="method", how="left")
    df["multi_objective_score"] = df["mae"] + lambda_fairness * df["fairness_gap"]
    return df.sort_values("multi_objective_score")
