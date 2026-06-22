#!/usr/bin/env python
"""
Build a stratified CHOCLO sample for offline evaluation.

Loads the official BenchmarkCHOCLO.csv from Hugging Face, draws N questions
per country (default 15 x 18 countries), balancing Category and Difficulty
within each country as evenly as possible.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Hashable, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHOCLO_MODULE_PATH = ROOT / "fairness_toolkit" / "choclo.py"


def _load_choclo_module():
    spec = importlib.util.spec_from_file_location("choclo_module", CHOCLO_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_choclo = _load_choclo_module()
CHOCLO_COLUMNS = _choclo.CHOCLO_COLUMNS
CHOCLO_HF_CSV = _choclo.CHOCLO_HF_CSV
LOCAL_SAMPLE_PATH = _choclo.LOCAL_SAMPLE_PATH
load_choclo = _choclo.load_choclo
normalize_country = _choclo.normalize_country

StratumKey = Tuple[Hashable, Hashable]


def _stratum_seed(random_state: int, pais: str, category: str, difficulty: str) -> int:
    return hash((random_state, pais, category, difficulty)) % (2**32 - 1)


def stratified_sample_country(
    country_df: pd.DataFrame,
    n: int = 15,
    random_state: int = 42,
    pais_label: str | None = None,
) -> pd.DataFrame:
    """
    Sample ``n`` rows from one country, balancing Category x Difficulty strata.

    If a stratum has fewer rows than its allocation, all available rows are taken
    and the remaining quota is redistributed to other strata without raising errors.
    """
    if len(country_df) <= n:
        return country_df.copy()

    pais_label = pais_label or normalize_country(country_df["Country"].iloc[0])
    strata = {
        (category, difficulty): group
        for (category, difficulty), group in country_df.groupby(
            ["Category", "Difficulty"], sort=True
        )
    }

    n_strata = len(strata)
    base = n // n_strata
    extra = n % n_strata

    # Prefer assigning the remainder to larger strata so targets stay feasible.
    ranked_keys = sorted(
        strata.keys(),
        key=lambda key: len(strata[key]),
        reverse=True,
    )

    targets = {key: base for key in strata}
    for key in ranked_keys[:extra]:
        targets[key] += 1

    selected_parts: list[pd.DataFrame] = []
    selected_indices: set[int] = set()

    for key, group in strata.items():
        take = min(targets[key], len(group))
        if take == 0:
            continue
        picked = group.sample(
            n=take,
            random_state=_stratum_seed(random_state, pais_label, key[0], key[1]),
        )
        selected_parts.append(picked)
        selected_indices.update(picked.index.tolist())

    shortfall = n - len(selected_indices)
    if shortfall > 0:
        remaining = country_df[~country_df.index.isin(selected_indices)]
        if not remaining.empty:
            extra_take = min(shortfall, len(remaining))
            extra_picked = remaining.sample(
                n=extra_take,
                random_state=_stratum_seed(random_state, pais_label, "__extra__", shortfall),
            )
            selected_parts.append(extra_picked)

    if not selected_parts:
        return country_df.sample(n=n, random_state=random_state)

    sample = pd.concat(selected_parts).drop_duplicates()
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=random_state)

    return sample


def build_stratified_sample(
    df: pd.DataFrame,
    n_per_country: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a stratified sample with ``n_per_country`` rows for each country."""
    working = df.copy()
    working["_pais"] = working["Country"].map(normalize_country)

    country_parts: list[pd.DataFrame] = []
    for pais, country_df in working.groupby("_pais", sort=True):
        country_sample = stratified_sample_country(
            country_df,
            n=n_per_country,
            random_state=random_state,
            pais_label=pais,
        )
        country_parts.append(country_sample)

    sample = pd.concat(country_parts, ignore_index=False)
    return sample[CHOCLO_COLUMNS].reset_index(drop=True)


def summarize_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact summary of sample balance by country and stratum."""
    summary = df.copy()
    summary["pais"] = summary["Country"].map(normalize_country)
    return (
        summary.groupby(["pais", "Category", "Difficulty"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["pais", "Category", "Difficulty"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a stratified CHOCLO sample CSV for offline evaluation."
    )
    parser.add_argument(
        "--n-per-country",
        type=int,
        default=15,
        help="Number of questions to sample per country (default: 15).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LOCAL_SAMPLE_PATH,
        help=f"Output CSV path (default: {LOCAL_SAMPLE_PATH}).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=CHOCLO_HF_CSV,
        help="Source CSV path or Hugging Face URL.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a balance summary after writing the sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading CHOCLO from {args.source} ...")
    df = load_choclo(args.source)
    print(f"Loaded {len(df):,} rows.")

    sample = build_stratified_sample(
        df,
        n_per_country=args.n_per_country,
        random_state=args.random_state,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False, encoding="utf-8")

    n_countries = sample["Country"].map(normalize_country).nunique()
    print(
        f"Saved {len(sample):,} rows "
        f"({args.n_per_country} x {n_countries} countries) to {args.output}"
    )

    if args.summary:
        summary = summarize_sample(sample)
        print("\nSample balance by country / category / difficulty:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
