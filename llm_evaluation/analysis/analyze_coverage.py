#!/usr/bin/env python
"""
Analyze thematic coverage of CHOCLO questions via embedding space geometry.

Loads CHOCLO questions, embeds them with the same multilingual model used in
evaluation, reduces to 2D with UMAP, and summarizes per-country dispersion as
a proxy for thematic diversity.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "coverage"
EMBEDDINGS_MODULE_PATH = ROOT / "fairness_toolkit" / "embeddings.py"
CHOCLO_MODULE_PATH = ROOT / "fairness_toolkit" / "choclo.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_choclo = _load_module(CHOCLO_MODULE_PATH, "choclo_module")
_embeddings = _load_module(EMBEDDINGS_MODULE_PATH, "embeddings_module")

CHOCLO_HF_CSV = _choclo.CHOCLO_HF_CSV
LOCAL_SAMPLE_PATH = _choclo.LOCAL_SAMPLE_PATH
load_choclo = _choclo.load_choclo
normalize_country = _choclo.normalize_country
SemanticSimilarityScorer = _embeddings.SemanticSimilarityScorer

DEFAULT_COVERAGE_SAMPLE_DIR = ROOT / "data" / "coverage_sample"
DEFAULT_COVERAGE_FULL_DIR = ROOT / "data" / "coverage_full"

MODE_PRESETS = {
    "sample": {
        "source": str(LOCAL_SAMPLE_PATH),
        "n": 0,
        "output_dir": DEFAULT_COVERAGE_SAMPLE_DIR,
    },
    "full": {
        "source": CHOCLO_HF_CSV,
        "n": 5000,
        "output_dir": DEFAULT_COVERAGE_FULL_DIR,
    },
}


def apply_mode_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply standardized ``--mode sample|full`` settings when requested."""
    if not getattr(args, "mode", None):
        return args

    preset = MODE_PRESETS[args.mode]
    args.source = preset["source"]
    args.n = preset["n"]
    args.output_dir = preset["output_dir"]
    return args


def stratified_sample_by_country(
    df: pd.DataFrame,
    n: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Draw ``n`` rows with balanced representation across countries."""
    working = df.copy()
    working["pais"] = working["Country"].map(normalize_country)
    countries = sorted(working["pais"].unique())

    if n >= len(working):
        return working.drop(columns=["pais"]).reset_index(drop=True)

    base = n // len(countries)
    extra = n % len(countries)
    parts = []

    for index, pais in enumerate(countries):
        take = base + (1 if index < extra else 0)
        country_df = working[working["pais"] == pais]
        sampled = country_df.sample(
            n=min(take, len(country_df)),
            random_state=random_state + index,
        )
        parts.append(sampled)

    sample = pd.concat(parts, ignore_index=True)
    return sample.drop(columns=["pais"]).reset_index(drop=True)


def embed_questions(
    questions: list[str],
    model_name: str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode questions with the evaluation embedding model."""
    scorer = SemanticSimilarityScorer(model_name=model_name)
    embeddings = scorer.model.encode(
        questions,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    return np.asarray(embeddings)


def reduce_umap(
    embeddings: np.ndarray,
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Project embeddings to 2D with UMAP."""
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def compute_country_coverage(
    coords_2d: np.ndarray,
    paises: np.ndarray,
) -> pd.DataFrame:
    """
    Compute per-country dispersion in reduced space.

    ``dispersion_promedio`` is the mean standard deviation across UMAP axes.
    Higher values suggest broader thematic spread in question embedding space.
    """
    rows = []
    for pais in sorted(np.unique(paises)):
        mask = paises == pais
        points = coords_2d[mask]
        n_preguntas = int(len(points))

        if n_preguntas < 2:
            dispersion = 0.0
        else:
            dispersion = float(np.mean(np.std(points, axis=0)))

        rows.append(
            {
                "pais": pais,
                "n_preguntas": n_preguntas,
                "dispersion_promedio": dispersion,
            }
        )

    table = pd.DataFrame(rows)
    table = table.sort_values(
        "dispersion_promedio",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    table["ranking_cobertura"] = np.arange(1, len(table) + 1)
    return table


def plot_scatter(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Save a 2D scatter plot colored by categorical labels."""
    fig, ax = plt.subplots(figsize=(12, 8))
    unique_labels = sorted(pd.Series(labels).astype(str).unique())

    cmap = plt.get_cmap("tab20")
    if len(unique_labels) > 20:
        cmap = plt.get_cmap("nipy_spectral", len(unique_labels))

    for index, label in enumerate(unique_labels):
        mask = labels.astype(str) == label
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            s=12,
            alpha=0.65,
            color=cmap(index % cmap.N),
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
        markerscale=2,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze CHOCLO question coverage in embedding space."
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_PRESETS),
        default=None,
        help=(
            "Standard preset: 'sample' = local choclo_sample.csv (same as run_all); "
            "'full' = HF CHOCLO stratified n=5000. Omit to use manual flags below."
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        default=CHOCLO_HF_CSV,
        help="CHOCLO CSV path or Hugging Face URL.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5000,
        help="Number of rows to analyze (stratified by country). Use 0 for full dataset.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling and UMAP (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64).",
    )
    args = parser.parse_args()
    return apply_mode_preset(args)


def main() -> None:
    args = parse_args()
    if args.mode:
        print(
            f"Using --mode {args.mode}: source={args.source}, n={args.n}, "
            f"output_dir={args.output_dir}"
        )
    output_dir = args.output_dir
    plots_dir = output_dir / "plots"

    print(f"Loading CHOCLO from {args.source} ...")
    df = load_choclo(args.source)
    print(f"Loaded {len(df):,} rows.")

    if args.n and args.n > 0 and args.n < len(df):
        df = stratified_sample_by_country(df, n=args.n, random_state=args.random_state)
        print(f"Using stratified sample of {len(df):,} rows.")
    else:
        print("Using full dataset.")

    df = df.copy()
    df["pais"] = df["Country"].map(normalize_country)
    questions = df["Question"].astype(str).tolist()

    print("Generating question embeddings ...")
    embeddings = embed_questions(questions, batch_size=args.batch_size)

    print("Running UMAP ...")
    coords_2d = reduce_umap(embeddings, random_state=args.random_state)

    print("Computing coverage by country ...")
    coverage = compute_country_coverage(coords_2d, df["pais"].to_numpy())

    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = output_dir / "coverage_by_country.csv"
    coverage.to_csv(coverage_path, index=False, encoding="utf-8")

    print("Saving scatter plots ...")
    plot_scatter(
        coords_2d,
        df["pais"].astype(str).to_numpy(),
        title="CHOCLO questions in embedding space (colored by country)",
        output_path=plots_dir / "scatter_by_country.png",
    )
    plot_scatter(
        coords_2d,
        df["Category"].astype(str).to_numpy(),
        title="CHOCLO questions in embedding space (colored by Category)",
        output_path=plots_dir / "scatter_by_category.png",
    )

    coords_df = df[["Entity", "Country", "Category", "Difficulty", "pais"]].copy()
    coords_df["umap_x"] = coords_2d[:, 0]
    coords_df["umap_y"] = coords_2d[:, 1]
    coords_df.to_csv(output_dir / "umap_coordinates.csv", index=False, encoding="utf-8")

    print(f"\nSaved coverage table to {coverage_path}")
    print(f"Saved plots to {plots_dir}")
    print("\nTop countries by thematic dispersion:")
    print(
        coverage[["ranking_cobertura", "pais", "n_preguntas", "dispersion_promedio"]]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
