#!/usr/bin/env python
"""
Cross category-level MAE with country composition and intersectional groups.

Hypothesis: part of reported country-level bias may reflect uneven category
mix (e.g. more public_figure/tradition in some countries) rather than pure
geographic bias.

Outputs (under results dir):
  - category_composition_analysis.csv  per pais x model
  - category_mae_by_country.csv        pais x category x model
  - category_difficulty_summary.csv      global ranking + consistency checks
  - interseccion_by_category.csv         intersection MAE rolled up by category
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from results_io import DEFAULT_OUTPUT_DIR  # noqa: E402

DEFAULT_HARD_CATEGORIES = ("public_figure", "tradition")
DEFAULT_EASY_CATEGORIES = ("geography", "flora")
DEFAULT_MODELS = ("GPT", "Claude", "LatamGPT")


def _valid_similarity_column(model: str) -> str:
    return f"similarity_{model}"


def _error_column(model: str) -> str:
    return f"error_{model}"


def _models_in_scored_df(scored_df: pd.DataFrame) -> list[str]:
    models = []
    for col in scored_df.columns:
        if col.startswith("similarity_"):
            models.append(col.removeprefix("similarity_"))
    return models


def compute_category_mae_by_country(
    scored_df: pd.DataFrame,
    model_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """MAE (1 - similarity) for each pais x Category x model."""
    model_names = model_names or _models_in_scored_df(scored_df)
    rows: list[dict] = []

    for model in model_names:
        sim_col = _valid_similarity_column(model)
        err_col = _error_column(model)
        if sim_col not in scored_df.columns:
            continue

        subset = scored_df[scored_df[sim_col].notna()].copy()
        if subset.empty:
            continue

        if err_col not in subset.columns:
            subset[err_col] = 1.0 - subset[sim_col]

        grouped = (
            subset.groupby(["pais", "Category"], sort=True)[err_col]
            .agg(mae="mean", n_preguntas="count")
            .reset_index()
        )
        grouped["method"] = model
        rows.extend(grouped.to_dict(orient="records"))

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result = result.rename(columns={"Category": "category"})
    return result.sort_values(["method", "pais", "category"]).reset_index(drop=True)


def summarize_category_difficulty(
    df_mae_by_category: pd.DataFrame,
    df_category_by_country: pd.DataFrame,
    hard_categories: Sequence[str] = DEFAULT_HARD_CATEGORIES,
    easy_categories: Sequence[str] = DEFAULT_EASY_CATEGORIES,
) -> pd.DataFrame:
    """Global category ranking plus cross-country consistency flags."""
    rows: list[dict] = []

    for method in sorted(df_mae_by_category["method"].unique()):
        global_cat = df_mae_by_category[df_mae_by_category["method"] == method].copy()
        by_country = df_category_by_country[df_category_by_country["method"] == method]

        countries = sorted(by_country["pais"].unique()) if not by_country.empty else []

        for _, row in global_cat.iterrows():
            category = row["category"]
            entry = {
                "method": method,
                "category": category,
                "mae_global": row["mae"],
                "n_preguntas_global": row["n_preguntas"],
                "confiable": row.get("confiable"),
                "is_hard_category": category in hard_categories,
                "is_easy_category": category in easy_categories,
            }

            for easy_cat in easy_categories:
                if by_country.empty or easy_cat not in global_cat["category"].values:
                    entry[f"higher_than_{easy_cat}_all_countries"] = None
                    continue

                higher_in_all = True
                comparable_countries = 0
                for pais in countries:
                    cat_mae = by_country[
                        (by_country["pais"] == pais) & (by_country["category"] == category)
                    ]
                    easy_mae = by_country[
                        (by_country["pais"] == pais) & (by_country["category"] == easy_cat)
                    ]
                    if cat_mae.empty or easy_mae.empty:
                        continue
                    comparable_countries += 1
                    if float(cat_mae["mae"].iloc[0]) <= float(easy_mae["mae"].iloc[0]):
                        higher_in_all = False
                        break

                entry[f"higher_than_{easy_cat}_all_countries"] = (
                    higher_in_all if comparable_countries > 0 else None
                )

            rows.append(entry)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    return summary.sort_values(["method", "mae_global"], ascending=[True, False]).reset_index(
        drop=True
    )


def _category_weights(scored_df: pd.DataFrame, model: str) -> tuple[pd.Series, pd.Series]:
    """Return global and per-pais category proportions for valid model rows."""
    sim_col = _valid_similarity_column(model)
    subset = scored_df[scored_df[sim_col].notna()].copy()
    global_weights = subset["Category"].value_counts(normalize=True)
    by_pais = (
        subset.groupby("pais")["Category"]
        .value_counts(normalize=True)
        .rename("share")
        .reset_index()
    )
    return global_weights, by_pais


def build_country_composition_analysis(
    scored_df: pd.DataFrame,
    df_group_mae_pais: pd.DataFrame,
    df_mae_by_category: pd.DataFrame,
    hard_categories: Sequence[str] = DEFAULT_HARD_CATEGORIES,
) -> pd.DataFrame:
    """
    Compare observed country MAE with MAE expected from each country's category mix.
    """
    rows: list[dict] = []

    for method in sorted(df_group_mae_pais["method"].unique()):
        pais_mae = df_group_mae_pais[df_group_mae_pais["method"] == method]
        cat_mae = df_mae_by_category[df_mae_by_category["method"] == method].set_index(
            "category"
        )["mae"]
        global_weights, by_pais = _category_weights(scored_df, method)

        global_hard_share = float(global_weights.reindex(hard_categories, fill_value=0.0).sum())

        for _, pais_row in pais_mae.iterrows():
            pais = pais_row["pais"]
            observed = float(pais_row["mae"])

            mix = by_pais[by_pais["pais"] == pais]
            if mix.empty:
                expected = np.nan
                hard_share = np.nan
                hard_overrep = np.nan
                top_overrep_category = None
                top_overrep_delta = np.nan
            else:
                expected = 0.0
                weight_sum = 0.0
                overrep_rows: list[tuple[str, float]] = []

                for _, mix_row in mix.iterrows():
                    category = mix_row["Category"]
                    share = float(mix_row["share"])
                    global_share = float(global_weights.get(category, 0.0))
                    overrep_rows.append((category, share - global_share))
                    if category in cat_mae.index:
                        expected += share * float(cat_mae[category])
                        weight_sum += share

                expected = expected / weight_sum if weight_sum else np.nan
                hard_share = float(
                    mix[mix["Category"].isin(hard_categories)]["share"].sum()
                )
                hard_overrep = hard_share - global_hard_share

                overrep_rows.sort(key=lambda item: item[1], reverse=True)
                top_overrep_category, top_overrep_delta = overrep_rows[0]

            residual = observed - expected if pd.notna(expected) else np.nan

            rows.append(
                {
                    "pais": pais,
                    "method": method,
                    "mae_observed": observed,
                    "mae_expected_from_category_mix": expected,
                    "mae_composition_residual": residual,
                    "n_preguntas": int(pais_row["n_preguntas"]),
                    "confiable_pais": pais_row.get("confiable"),
                    "hard_category_share": hard_share,
                    "hard_category_share_global": global_hard_share,
                    "hard_category_overrepresentation": hard_overrep,
                    "top_overrepresented_category": top_overrep_category,
                    "top_overrepresentation_delta": top_overrep_delta,
                    "interpretacion": _interpret_composition_row(
                        residual, hard_overrep, top_overrep_category, hard_categories
                    ),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    return result.sort_values(
        ["method", "mae_composition_residual"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)


def _interpret_composition_row(
    residual: float,
    hard_overrep: float,
    top_category: Optional[str],
    hard_categories: Sequence[str],
) -> Optional[str]:
    if pd.isna(residual) or pd.isna(hard_overrep):
        return None

    parts: list[str] = []
    if hard_overrep > 0.05:
        parts.append(
            "sobre-representacion de categorias dificiles "
            f"({', '.join(hard_categories)})"
        )
    elif hard_overrep < -0.05:
        parts.append("sub-representacion de categorias dificiles")

    if residual > 0.03:
        parts.append("MAE peor de lo que explica la mezcla de categorias")
    elif residual < -0.03:
        parts.append("MAE mejor de lo que explica la mezcla de categorias")

    if top_category and top_category in hard_categories:
        parts.append(f"categoria mas sobre-representada: {top_category}")

    return "; ".join(parts) if parts else "mezcla de categorias alineada con el promedio global"


def rollup_interseccion_by_category(
    df_group_mae_interseccion: pd.DataFrame,
    scored_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate intersection-level MAE by category using scored_df metadata."""
    lookup = (
        scored_df[["interseccion", "Category", "pais"]]
        .drop_duplicates(subset=["interseccion"])
        .rename(columns={"Category": "category"})
    )
    merged = df_group_mae_interseccion.merge(lookup, on="interseccion", how="left")
    merged = merged[merged["category"].notna()].copy()

    if merged.empty:
        return pd.DataFrame()

    grouped = (
        merged.groupby(["method", "category"], sort=True)
        .agg(
            mae_mean=("mae", "mean"),
            mae_median=("mae", "median"),
            n_cells=("mae", "count"),
            n_preguntas_total=("n_preguntas", "sum"),
            n_confiable=("confiable", "sum"),
        )
        .reset_index()
    )
    return grouped.sort_values(["method", "mae_mean"], ascending=[True, False]).reset_index(
        drop=True
    )


def worst_countries_category_profile(
    df_composition: pd.DataFrame,
    df_category_by_country: pd.DataFrame,
    df_mae_by_category: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Profile category mix for the worst countries by observed MAE."""
    rows: list[dict] = []

    for method in sorted(df_composition["method"].unique()):
        subset = df_composition[df_composition["method"] == method].sort_values(
            "mae_observed", ascending=False
        )
        global_cat_mae = df_mae_by_category[df_mae_by_category["method"] == method].set_index(
            "category"
        )["mae"]
        median_mae = float(global_cat_mae.median()) if not global_cat_mae.empty else np.nan

        for rank, (_, pais_row) in enumerate(subset.head(top_n).iterrows(), start=1):
            pais = pais_row["pais"]
            mix = df_category_by_country[
                (df_category_by_country["method"] == method)
                & (df_category_by_country["pais"] == pais)
            ]
            hard_cats_in_mix = mix[mix["category"].isin(DEFAULT_HARD_CATEGORIES)]
            hard_mae_values = [
                float(global_cat_mae.get(cat, np.nan))
                for cat in hard_cats_in_mix["category"]
                if cat in global_cat_mae.index
            ]

            rows.append(
                {
                    "method": method,
                    "pais_rank_by_mae": rank,
                    "pais": pais,
                    "mae_observed": pais_row["mae_observed"],
                    "mae_composition_residual": pais_row["mae_composition_residual"],
                    "hard_category_overrepresentation": pais_row[
                        "hard_category_overrepresentation"
                    ],
                    "n_hard_category_questions": int(hard_cats_in_mix["n_preguntas"].sum())
                    if not hard_cats_in_mix.empty
                    else 0,
                    "mean_global_mae_of_hard_categories_present": float(np.nanmean(hard_mae_values))
                    if hard_mae_values
                    else np.nan,
                    "global_median_category_mae": median_mae,
                }
            )

    return pd.DataFrame(rows)


def run_category_composition_analysis(
    scored_df: pd.DataFrame,
    df_group_mae_pais: pd.DataFrame,
    df_mae_by_category: pd.DataFrame,
    df_group_mae_interseccion: Optional[pd.DataFrame] = None,
    hard_categories: Sequence[str] = DEFAULT_HARD_CATEGORIES,
    easy_categories: Sequence[str] = DEFAULT_EASY_CATEGORIES,
) -> dict[str, pd.DataFrame]:
    """Run all category-composition analyses and return output tables."""
    df_category_by_country = compute_category_mae_by_country(scored_df)
    df_difficulty = summarize_category_difficulty(
        df_mae_by_category,
        df_category_by_country,
        hard_categories=hard_categories,
        easy_categories=easy_categories,
    )
    df_composition = build_country_composition_analysis(
        scored_df,
        df_group_mae_pais,
        df_mae_by_category,
        hard_categories=hard_categories,
    )
    df_worst_profile = worst_countries_category_profile(
        df_composition,
        df_category_by_country,
        df_mae_by_category,
    )

    outputs: dict[str, pd.DataFrame] = {
        "category_mae_by_country.csv": df_category_by_country,
        "category_difficulty_summary.csv": df_difficulty,
        "category_composition_analysis.csv": df_composition,
        "worst_countries_category_profile.csv": df_worst_profile,
    }

    if df_group_mae_interseccion is not None and not df_group_mae_interseccion.empty:
        outputs["interseccion_by_category.csv"] = rollup_interseccion_by_category(
            df_group_mae_interseccion,
            scored_df,
        )

    return outputs


def load_results_from_dir(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load saved evaluation outputs needed for offline analysis."""
    paths = {
        "scored_df": results_dir / "evaluation_results.csv",
        "df_group_mae_pais": results_dir / "group_mae_pais.csv",
        "df_mae_by_category": results_dir / "mae_by_category.csv",
        "df_group_mae_interseccion": results_dir / "group_mae_interseccion.csv",
    }
    loaded: dict[str, pd.DataFrame] = {}
    for key, path in paths.items():
        if path.exists():
            loaded[key] = pd.read_csv(path)

    if "scored_df" in loaded and "df_mae_by_category" not in loaded:
        loaded["df_mae_by_category"] = _build_mae_by_category_from_scored(
            loaded["scored_df"]
        )
        loaded["_rebuilt_mae_by_category"] = loaded["df_mae_by_category"]

    return loaded


def _build_mae_by_category_from_scored(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild category MAE table from evaluation_results.csv when missing."""
    from fairness_toolkit.group_mae_stats import build_group_mae_category_tables

    preds_dict = {}
    for col in scored_df.columns:
        if col.startswith("similarity_"):
            model = col.removeprefix("similarity_")
            preds_dict[model] = scored_df[col].to_numpy(dtype=float)

    if not preds_dict:
        return pd.DataFrame()

    y_true = scored_df["y_true"].to_numpy(dtype=float)
    category = scored_df["Category"].astype(str).to_numpy()
    return build_group_mae_category_tables(
        y_true,
        preds_dict,
        category,
        "CHOCLO",
        min_n=1,
        n_bootstrap=100,
    )


def save_analysis_outputs(outputs: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for filename, df in outputs.items():
        if df is None or df.empty:
            continue
        path = output_dir / filename
        df.to_csv(path, index=False, encoding="utf-8")
        saved[filename] = path
    return saved


def print_analysis_summary(outputs: dict[str, pd.DataFrame]) -> None:
    summary = outputs.get("category_difficulty_summary.csv")
    composition = outputs.get("category_composition_analysis.csv")

    if summary is not None and not summary.empty:
        print("\n--- Dificultad global por categoria ---")
        for method in summary["method"].unique():
            subset = summary[summary["method"] == method]
            print(f"\n{method}:")
            for _, row in subset.iterrows():
                flags = []
                for col in row.index:
                    if col.startswith("higher_than_") and col.endswith("_all_countries"):
                        if row[col] is True:
                            flags.append(col.replace("higher_than_", "").replace("_all_countries", ""))
                flag_text = f" > {', '.join(flags)} en todos los paises" if flags else ""
                print(
                    f"  {row['category']}: MAE={row['mae_global']:.3f} "
                    f"(n={row['n_preguntas_global']}){flag_text}"
                )

    if composition is not None and not composition.empty:
        print("\n--- Paises con mayor residual de composicion (MAE peor que mezcla) ---")
        for method in composition["method"].unique():
            subset = composition[composition["method"] == method].head(5)
            print(f"\n{method}:")
            for _, row in subset.iterrows():
                print(
                    f"  {row['pais']}: observado={row['mae_observed']:.3f}, "
                    f"esperado={row['mae_expected_from_category_mix']:.3f}, "
                    f"residual={row['mae_composition_residual']:.3f} — {row['interpretacion']}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether country-level MAE bias reflects category composition."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory with evaluation_results.csv and group MAE tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write analysis CSVs (default: same as --results-dir).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.results_dir

    loaded = load_results_from_dir(args.results_dir)
    required = ("scored_df", "df_group_mae_pais", "df_mae_by_category")
    missing = [key for key in required if key not in loaded]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {args.results_dir}: {', '.join(missing)}. "
            "Run run_all.py first."
        )

    outputs = run_category_composition_analysis(
        loaded["scored_df"],
        loaded["df_group_mae_pais"],
        loaded["df_mae_by_category"],
        loaded.get("df_group_mae_interseccion"),
    )
    if loaded.get("_rebuilt_mae_by_category") is not None:
        outputs["mae_by_category.csv"] = loaded["_rebuilt_mae_by_category"]
    saved = save_analysis_outputs(outputs, output_dir)
    print_analysis_summary(outputs)
    print(f"\nSaved {len(saved)} analysis file(s) to {output_dir.resolve()}")
    for name in sorted(saved):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
