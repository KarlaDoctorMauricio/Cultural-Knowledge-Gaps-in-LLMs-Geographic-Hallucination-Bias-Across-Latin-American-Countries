"""Save CHOCLO evaluation outputs to CSV and JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "data" / "results"


def _save_df(df, path: Path) -> None:
    if df is None or df.empty:
        return
    df.to_csv(path, index=False, encoding="utf-8")


def save_results(results: dict, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    file_map = {
        "evaluation_results.csv": results.get("scored_df"),
        "metrics_summary.csv": results.get("df_metrics"),
        "metrics_summary_post.csv": results.get("df_metrics_post"),
        "group_mae_pais.csv": results.get("df_group_mae_pais"),
        "group_mae_interseccion.csv": results.get("df_group_mae_interseccion"),
        "mae_by_category.csv": results.get("df_group_mae_category"),
        "group_mae_pais_post.csv": results.get("df_group_mae_pais_post"),
        "group_mae_interseccion_post.csv": results.get("df_group_mae_interseccion_post"),
        "mae_by_category_post.csv": results.get("df_group_mae_category_post"),
        "model_selection.csv": results.get("df_selection"),
        "response_quality_breakdown.csv": results.get("response_quality_breakdown"),
        "hallucination_by_country.csv": results.get("response_quality_by_country"),
        "low_score_classifications.csv": results.get("low_score_classifications"),
    }

    for filename, dataframe in file_map.items():
        path = output_dir / filename
        _save_df(dataframe, path)
        if path.exists():
            saved[filename] = path

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_name": results.get("config_name"),
        "best_method": results.get("best_method"),
        "available_models": results.get("available_models"),
        "unavailable_models": results.get("unavailable_models"),
        "latamgpt_note": results.get("latamgpt_note"),
        "output_files": {name: str(path) for name, path in saved.items()},
    }

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    saved["run_summary.json"] = summary_path
    return saved
