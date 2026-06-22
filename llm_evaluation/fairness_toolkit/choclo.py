"""CHOCLO dataset loading and fairness grouping columns."""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Optional, Union

import pandas as pd

CHOCLO_COLUMNS = [
    "Entity",
    "Country",
    "Category",
    "Difficulty",
    "Question",
    "Answer",
]

CHOCLO_HF_DATASET = "latam-gpt/CHOCLO"
CHOCLO_HF_CSV = f"hf://datasets/{CHOCLO_HF_DATASET}/BenchmarkCHOCLO.csv"
LOCAL_SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "choclo_sample.csv"


def load_choclo(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the CHOCLO benchmark.

    By default reads the official Hugging Face dataset:
    https://huggingface.co/datasets/latam-gpt/CHOCLO

        df = pd.read_csv("hf://datasets/latam-gpt/CHOCLO/BenchmarkCHOCLO.csv")

    Pass ``LOCAL_SAMPLE_PATH`` or another CSV path for offline runs/tests.
    """
    data_path = CHOCLO_HF_CSV if path is None else str(path)
    df = pd.read_csv(data_path)
    missing = [col for col in CHOCLO_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CHOCLO dataset missing columns: {missing}")
    return df[CHOCLO_COLUMNS].copy()


def sample_choclo(
    df: pd.DataFrame,
    n: Optional[int] = None,
    frac: Optional[float] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a random subset of CHOCLO rows."""
    if n is not None and frac is not None:
        raise ValueError("Specify either n or frac, not both.")
    if n is not None:
        return df.sample(n=min(n, len(df)), random_state=random_state).reset_index(drop=True)
    if frac is not None:
        return df.sample(frac=frac, random_state=random_state).reset_index(drop=True)
    return df.reset_index(drop=True)


def normalize_country(country: str) -> str:
    """Normalize country labels for consistent fairness grouping."""
    text = unicodedata.normalize("NFKD", str(country).strip().lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def add_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fairness grouping columns used by the existing toolkit.

    - pais: primary sensitive attribute (Country, normalized)
    - interseccion: Country_Category_Difficulty
    """
    result = df.copy()
    result["pais"] = result["Country"].map(normalize_country)
    result["interseccion"] = (
        result["pais"]
        + "_"
        + result["Category"].astype(str)
        + "_"
        + result["Difficulty"].astype(str)
    )
    return result
