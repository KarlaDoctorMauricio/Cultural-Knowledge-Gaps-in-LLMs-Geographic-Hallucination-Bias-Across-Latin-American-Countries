#!/usr/bin/env python
"""
Decompose country-level hallucination rates by category composition.

Mirrors ``analyze_category_composition.py`` (MAE mix vs residual) but uses
``pct_alucinacion`` from the LLM judge as the outcome.

Outputs:
  - alucinacion_by_category_global.csv   global % alucinacion per category x model
  - alucinacion_composition_residual.csv per pais x model with expected mix + residual
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.analyze_category_composition import (  # noqa: E402
    DEFAULT_HARD_CATEGORIES,
    _category_weights,
    _interpret_composition_row,
)
from evaluator import JUDGE_PROMPT_VERSION  # noqa: E402
from results_io import DEFAULT_OUTPUT_DIR  # noqa: E402

DEFAULT_JUDGE_MODELS = ("GPT", "Claude")
DEFAULT_CLASSIFICATIONS_FILE = "judge_final_results.csv"
LEGACY_CLASSIFICATIONS_FILE = "low_score_classifications.csv"
OUTPUT_FILENAME = "alucinacion_composition_residual.csv"
GLOBAL_CATEGORY_FILENAME = "alucinacion_by_category_global.csv"


def ensure_category_column(
    classified: pd.DataFrame,
    scored_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Ensure ``Category`` is present; merge from evaluation_results if needed."""
    if "Category" in classified.columns and classified["Category"].notna().all():
        return classified.copy()

    if scored_df is None:
        raise ValueError(
            "Classifications missing Category and no scored_df provided to merge from."
        )

    meta = scored_df.reset_index().rename(columns={"index": "row_index"})
    if "row_index" not in meta.columns:
        meta = meta.reset_index().rename(columns={"index": "row_index"})

    merge_cols = ["row_index", "Category"]
    if "Category" not in meta.columns:
        raise ValueError("scored_df has no Category column for merge.")

    merged = classified.merge(
        meta[merge_cols].drop_duplicates(subset=["row_index"]),
        on="row_index",
        how="left",
        suffixes=("", "_eval"),
    )
    if "Category_eval" in merged.columns:
        merged["Category"] = merged["Category"].fillna(merged["Category_eval"])
        merged = merged.drop(columns=["Category_eval"])

    if merged["Category"].isna().any():
        raise ValueError("Some classifications could not be matched to Category.")

    return merged


def compute_alucinacion_by_category_global(
    classified: pd.DataFrame,
    model_names: Sequence[str] = DEFAULT_JUDGE_MODELS,
) -> pd.DataFrame:
    """Global average % alucinacion for each CHOCLO category, pooled across countries."""
    if classified.empty or "judge_label" not in classified.columns:
        return pd.DataFrame(
            columns=["model", "category", "pct_alucinacion", "n_preguntas"]
        )

    working = classified[classified["model"].isin(model_names)].copy()
    rows: list[dict] = []

    for model in model_names:
        subset = working[working["model"] == model]
        if subset.empty:
            continue

        for category, group in subset.groupby("Category", sort=True):
            labels = group["judge_label"].astype(str)
            n_total = int(len(labels))
            n_aluc = int((labels == "alucinacion").sum())
            rows.append(
                {
                    "model": model,
                    "category": str(category),
                    "pct_alucinacion": round(100.0 * n_aluc / n_total, 2) if n_total else np.nan,
                    "n_preguntas": n_total,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["model", "category", "pct_alucinacion", "n_preguntas"])

    return pd.DataFrame(rows).sort_values(["model", "pct_alucinacion"], ascending=[True, False])


def build_alucinacion_composition_analysis(
    classified: pd.DataFrame,
    proportions_by_country: pd.DataFrame,
    scored_df: pd.DataFrame,
    df_alucinacion_by_category: pd.DataFrame,
    hard_categories: Sequence[str] = DEFAULT_HARD_CATEGORIES,
) -> pd.DataFrame:
    """
    Compare observed country hallucination rate with rate expected from category mix.

    Uses the same per-country category weights as ``category_composition_analysis.csv``.
    """
    if proportions_by_country.empty:
        return pd.DataFrame()

    rows: list[dict] = []

    for model in sorted(proportions_by_country["model"].unique()):
        prop_subset = proportions_by_country[proportions_by_country["model"] == model]
        cat_aluc = df_alucinacion_by_category[
            df_alucinacion_by_category["model"] == model
        ].set_index("category")["pct_alucinacion"]

        try:
            global_weights, by_pais = _category_weights(scored_df, model)
        except KeyError:
            continue

        global_hard_share = float(global_weights.reindex(hard_categories, fill_value=0.0).sum())

        for _, pais_row in prop_subset.iterrows():
            pais = pais_row["pais"]
            observed = float(pais_row["pct_alucinacion"])

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
                    if category in cat_aluc.index:
                        expected += share * float(cat_aluc[category])
                        weight_sum += share

                expected = expected / weight_sum if weight_sum else np.nan
                hard_share = float(
                    mix[mix["Category"].isin(hard_categories)]["share"].sum()
                )
                hard_overrep = hard_share - global_hard_share

                overrep_rows.sort(key=lambda item: item[1], reverse=True)
                top_overrep_category, top_overrep_delta = overrep_rows[0]

            residual = observed - expected if pd.notna(expected) else np.nan
            residual_pp = round(residual, 2) if pd.notna(residual) else np.nan

            rows.append(
                {
                    "pais": pais,
                    "model": model,
                    "pct_alucinacion_observado": observed,
                    "pct_alucinacion_esperado_por_mezcla": round(expected, 2)
                    if pd.notna(expected)
                    else np.nan,
                    "residual_alucinacion": residual_pp,
                    "n_total": int(pais_row.get("n_total", np.nan))
                    if pd.notna(pais_row.get("n_total"))
                    else np.nan,
                    "confiable": pais_row.get("confiable"),
                    "hard_category_share": hard_share,
                    "hard_category_share_global": global_hard_share,
                    "hard_category_overrepresentation": hard_overrep,
                    "top_overrepresented_category": top_overrep_category,
                    "top_overrepresentation_delta": top_overrep_delta,
                    "interpretacion": _interpret_alucinacion_row(
                        residual_pp, hard_overrep, top_overrep_category, hard_categories
                    ),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    return result.sort_values(
        ["model", "residual_alucinacion"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)


def _interpret_alucinacion_row(
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

    if residual > 3.0:
        parts.append(
            "alucinacion peor de lo que explica la mezcla de categorias"
        )
    elif residual < -3.0:
        parts.append(
            "alucinacion mejor de lo que explica la mezcla de categorias"
        )

    if top_category and top_category in hard_categories:
        parts.append(f"categoria mas sobre-representada: {top_category}")

    return (
        "; ".join(parts)
        if parts
        else "mezcla de categorias alineada con el promedio global"
    )


def run_alucinacion_composition_analysis(
    classified: pd.DataFrame,
    proportions_by_country: pd.DataFrame,
    scored_df: pd.DataFrame,
    output_dir: Path,
    model_names: Sequence[str] = DEFAULT_JUDGE_MODELS,
    hard_categories: Sequence[str] = DEFAULT_HARD_CATEGORIES,
) -> dict[str, pd.DataFrame]:
    """Run hallucination composition decomposition and save CSVs."""
    working = ensure_category_column(classified, scored_df)
    df_global = compute_alucinacion_by_category_global(working, model_names=model_names)
    df_composition = build_alucinacion_composition_analysis(
        working,
        proportions_by_country,
        scored_df,
        df_global,
        hard_categories=hard_categories,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, pd.DataFrame] = {
        GLOBAL_CATEGORY_FILENAME: df_global,
        OUTPUT_FILENAME: df_composition,
    }

    for filename, df in saved.items():
        if df is not None and not df.empty:
            df.to_csv(output_dir / filename, index=False, encoding="utf-8")

    return saved


def print_alucinacion_composition_summary(outputs: dict[str, pd.DataFrame]) -> None:
    """Print gap comparison: observed vs mix-expected vs residual."""
    df = outputs.get(OUTPUT_FILENAME)
    global_df = outputs.get(GLOBAL_CATEGORY_FILENAME)

    if global_df is not None and not global_df.empty:
        print("\n--- % alucinacion global por categoria ---")
        for model in global_df["model"].unique():
            subset = global_df[global_df["model"] == model].head(5)
            print(f"\n{model} (top 5 categorias):")
            for _, row in subset.iterrows():
                print(
                    f"  {row['category']}: {row['pct_alucinacion']:.1f}% "
                    f"(n={row['n_preguntas']})"
                )

    if df is None or df.empty:
        return

    print("\n--- Gap de alucinacion: observado vs explicado por mezcla ---")
    for model in df["model"].unique():
        subset = df[df["model"] == model]
        bias_obs = float(subset["pct_alucinacion_observado"].max() - subset["pct_alucinacion_observado"].min())
        bias_exp = float(
            subset["pct_alucinacion_esperado_por_mezcla"].max()
            - subset["pct_alucinacion_esperado_por_mezcla"].min()
        )
        bias_res = float(subset["residual_alucinacion"].max() - subset["residual_alucinacion"].min())
        print(
            f"  {model}: gap observado={bias_obs:.1f} pp, "
            f"gap esperado por mezcla={bias_exp:.1f} pp, "
            f"gap residual={bias_res:.1f} pp"
        )

    print("\n--- Paises con mayor residual de alucinacion (peor que mezcla) ---")
    for model in df["model"].unique():
        subset = df[df["model"] == model].head(5)
        print(f"\n{model}:")
        for _, row in subset.iterrows():
            print(
                f"  {row['pais']}: observado={row['pct_alucinacion_observado']:.1f}%, "
                f"esperado={row['pct_alucinacion_esperado_por_mezcla']:.1f}%, "
                f"residual={row['residual_alucinacion']:.1f} pp — {row['interpretacion']}"
            )


def resolve_classifications_path(
    results_dir: Path,
    *,
    explicit_path: Path | None = None,
    expected_version: str = JUDGE_PROMPT_VERSION,
) -> Path:
    """
    Prefer ``judge_final_results.csv`` (post-stability v2 labels).

    Falls back to ``low_score_classifications.csv`` only if the final file
    is missing, and validates ``judge_prompt_version`` when present.
    """
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Classifications file not found: {path}")
    else:
        final_path = results_dir / DEFAULT_CLASSIFICATIONS_FILE
        legacy_path = results_dir / LEGACY_CLASSIFICATIONS_FILE
        if final_path.exists():
            path = final_path
        elif legacy_path.exists():
            print(
                f"Warning: {DEFAULT_CLASSIFICATIONS_FILE} not found; "
                f"falling back to {LEGACY_CLASSIFICATIONS_FILE}."
            )
            path = legacy_path
        else:
            raise FileNotFoundError(
                f"Missing {DEFAULT_CLASSIFICATIONS_FILE} (and legacy "
                f"{LEGACY_CLASSIFICATIONS_FILE}) in {results_dir}."
            )

    sample = pd.read_csv(path, nrows=1)
    if "judge_prompt_version" in sample.columns:
        versions = pd.read_csv(path, usecols=["judge_prompt_version"])["judge_prompt_version"].dropna().unique()
        if len(versions) and expected_version not in versions:
            raise ValueError(
                f"{path.name} has judge_prompt_version={list(versions)}; "
                f"expected {expected_version!r}. Re-run rerun_judge_stability.py."
            )

    return path


def load_results_from_dir(
    results_dir: Path,
    *,
    classifications_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    classified_path = resolve_classifications_path(
        results_dir,
        explicit_path=classifications_path,
    )
    paths = {
        "classified": classified_path,
        "classified_path": classified_path,
        "proportions": results_dir / "category_proportions_by_country.csv",
        "scored_df": results_dir / "evaluation_results.csv",
    }
    loaded: dict[str, pd.DataFrame | Path] = {}
    for key, path in paths.items():
        if key == "classified_path":
            loaded[key] = path
            continue
        if Path(path).exists():
            loaded[key] = pd.read_csv(path)

    if "proportions" in loaded:
        props = loaded["proportions"]
        if "judge_source_file" in props.columns:
            sources = props["judge_source_file"].dropna().unique()
            if len(sources) and DEFAULT_CLASSIFICATIONS_FILE not in sources:
                print(
                    f"Warning: category_proportions_by_country.csv sourced from "
                    f"{list(sources)}; expected {DEFAULT_CLASSIFICATIONS_FILE}."
                )

    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether country-level hallucination gaps reflect category composition."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory with classifications, proportions, and evaluation_results.csv.",
    )
    parser.add_argument(
        "--classifications",
        type=Path,
        default=None,
        help=(
            f"Judge labels CSV (default: <results-dir>/{DEFAULT_CLASSIFICATIONS_FILE})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write output CSVs (default: same as --results-dir).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.results_dir

    loaded = load_results_from_dir(
        args.results_dir,
        classifications_path=args.classifications,
    )
    classified_path = loaded["classified_path"]
    print(f"Using judge labels: {Path(classified_path).resolve()}")

    required = ("classified", "proportions", "scored_df")
    missing = [key for key in required if key not in loaded]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {args.results_dir}: {', '.join(missing)}. "
            "Run run_all.py first."
        )

    outputs = run_alucinacion_composition_analysis(
        loaded["classified"],
        loaded["proportions"],
        loaded["scored_df"],
        output_dir,
    )
    print_alucinacion_composition_summary(outputs)
    print(f"\nSaved analysis to {output_dir.resolve()}")
    for name in (GLOBAL_CATEGORY_FILENAME, OUTPUT_FILENAME):
        path = output_dir / name
        if path.exists():
            print(f"  - {name}")


if __name__ == "__main__":
    main()
