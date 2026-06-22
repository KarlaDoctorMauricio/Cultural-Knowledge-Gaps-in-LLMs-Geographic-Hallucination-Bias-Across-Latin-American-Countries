"""Incremental checkpoint I/O for long CHOCLO evaluation runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

DEFAULT_CHECKPOINT_DIR_NAME = "checkpoints"
RESPONSES_CHECKPOINT_NAME = "responses_checkpoint.csv"
JUDGE_CHECKPOINT_NAME = "judge_checkpoint.csv"
MANIFEST_NAME = "manifest.json"


def default_checkpoint_dir(output_dir: Path) -> Path:
    return output_dir / DEFAULT_CHECKPOINT_DIR_NAME


def _manifest_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / MANIFEST_NAME


def _responses_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / RESPONSES_CHECKPOINT_NAME


def _judge_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / JUDGE_CHECKPOINT_NAME


def load_manifest(checkpoint_dir: Path) -> dict:
    path = _manifest_path(checkpoint_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(checkpoint_dir: Path, manifest: dict) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    _manifest_path(checkpoint_dir).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def init_run_manifest(
    checkpoint_dir: Path,
    *,
    n_rows: int,
    model_names: Iterable[str],
    judge_models: Iterable[str],
) -> dict:
    manifest = load_manifest(checkpoint_dir)
    manifest.update(
        {
            "n_rows": n_rows,
            "model_names": list(model_names),
            "judge_models": list(judge_models),
        }
    )
    save_manifest(checkpoint_dir, manifest)
    return manifest


def load_responses_checkpoint(
    checkpoint_dir: Path,
    df: pd.DataFrame,
    model_names: Iterable[str],
) -> pd.DataFrame:
    """Merge saved response columns into ``df`` when the checkpoint matches."""
    path = _responses_path(checkpoint_dir)
    if not path.exists():
        return df

    manifest = load_manifest(checkpoint_dir)
    if manifest.get("n_rows") not in (None, len(df)):
        return df

    checkpoint = pd.read_csv(path)
    if len(checkpoint) != len(df):
        return df

    result = df.copy()
    for model in model_names:
        response_col = f"response_{model}"
        note_col = f"note_{model}"
        if response_col in checkpoint.columns:
            result[response_col] = checkpoint[response_col]
        if note_col in checkpoint.columns:
            result[note_col] = checkpoint[note_col]
    return result


def save_responses_checkpoint(
    checkpoint_dir: Path,
    df: pd.DataFrame,
    *,
    n_rows: int,
    model_names: Iterable[str],
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _responses_path(checkpoint_dir)
    df.to_csv(path, index=False, encoding="utf-8")
    manifest = load_manifest(checkpoint_dir)
    manifest["n_rows"] = n_rows
    manifest["model_names"] = list(model_names)
    manifest["responses_saved_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(checkpoint_dir, manifest)
    return path


def judge_task_key(row_index: int, model: str) -> tuple[int, str]:
    return int(row_index), str(model)


def load_judge_checkpoint(checkpoint_dir: Path) -> pd.DataFrame:
    path = _judge_path(checkpoint_dir)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def judge_done_keys(classified: pd.DataFrame) -> set[tuple[int, str]]:
    if classified.empty:
        return set()
    return {
        judge_task_key(row.row_index, row.model)
        for row in classified.itertuples(index=False)
    }


def save_judge_checkpoint(
    checkpoint_dir: Path,
    classified: pd.DataFrame,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _judge_path(checkpoint_dir)
    classified.to_csv(path, index=False, encoding="utf-8")
    manifest = load_manifest(checkpoint_dir)
    manifest["judge_rows_saved"] = int(len(classified))
    manifest["judge_saved_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(checkpoint_dir, manifest)
    return path


def clear_checkpoints(checkpoint_dir: Path) -> None:
    if not checkpoint_dir.exists():
        return
    for name in (
        RESPONSES_CHECKPOINT_NAME,
        JUDGE_CHECKPOINT_NAME,
        MANIFEST_NAME,
    ):
        path = checkpoint_dir / name
        if path.exists():
            path.unlink()


def response_is_done(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip() != ""
