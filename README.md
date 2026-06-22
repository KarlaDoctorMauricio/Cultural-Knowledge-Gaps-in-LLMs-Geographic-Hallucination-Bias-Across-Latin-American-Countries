# LLM Fairness Evaluation on CHOCLO

This repository contains the code and results of a comparative evaluation of **GPT-4o-mini** (`gpt-4o-mini`) and **Claude Haiku 4.5** (`claude-haiku-4-5-20251001`) on the Latin American cultural benchmark [CHOCLO](https://huggingface.co/datasets/latam-gpt/CHOCLO).

Model IDs are configured in `llm_evaluation/clients.py` (or via `OPENAI_MODEL` / `ANTHROPIC_MODEL` in `.env`). The quality **judge** uses the same models: GPT as primary classifier and Claude as fallback.

We measured:

1. **Semantic similarity** between model responses and CHOCLO references (multilingual embeddings).
2. **Response quality** with an LLM judge in four categories: correct, partial, hallucination, abstention.
3. **Fairness by country** (MAE, bias, hallucination rates with confidence intervals).
4. **Thematic coverage** of the sample (dispersion in UMAP embedding space).
5. **Representativity Index (IR)** by country, combining observed hallucination and compositional residual.

**Evaluated sample:** 270 questions, 15 per country, **18 countries** (Brazil is not in this sample). Generation temperature **T = 0.2**. Statistical analyses (sampling, bootstrap, UMAP) use fixed seed **42**; API calls do not use `seed`.

---

## Repository layout

```
F-KADM/
├── README.md                 ← this document (single project guide)
├── .env.example              ← API key template (copy to .env)
└── llm_evaluation/
    ├── run_all.py            ← evaluation + report tables (step 6)
    ├── run_report.py         ← report tables only (no re-evaluation)
    ├── evaluator.py          ← LLM judge
    ├── clients.py            ← OpenAI / Anthropic API
    ├── fairness_toolkit/     ← pipeline (embeddings, MAE, calibration)
    ├── analysis/             ← post-hoc scripts (report tables)
    ├── scripts/              ← sampling and judge stability
    ├── tests/
    └── data/
        ├── choclo_sample.csv
        ├── results/          ← result CSVs
        └── coverage_sample/  ← UMAP, coverage, plots
```

All executable code lives under `llm_evaluation/`.

---

## Setup and execution

### 1. Dependencies

```powershell
cd llm_evaluation
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. API keys

At the repo root, copy `.env.example` to `.env`:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

### 3. Evaluation + report tables (one command)

```powershell
python run_all.py
```

Generates responses, similarities, fairness metrics, judge labels, **and at the end** report tables (IR, `alucinacion_country_full.csv`, UMAP). Resumes from checkpoint if interrupted.

Useful options:

```powershell
python run_all.py --skip-report          # evaluation only, no UMAP/IR
python run_all.py --skip-coverage        # report without UMAP (faster)
python run_all.py -n 30                  # quick test
```

### 4. Consolidated judge (98.3% stability) — run once

After fixing the judge prompt, run **once** before the final report:

```powershell
python scripts/rerun_judge_stability.py
```

Canonical output: `judge_final_results.csv` (majority vote over 3 runs). Then:

```powershell
python run_report.py
```

`run_report.py` regenerates **all** report tables from `judge_final_results.csv` (proportions, residual, IR, unified table, UMAP). It does not call GPT/Claude for new responses.

```powershell
python run_report.py --skip-coverage     # without UMAP (~1 min)
```

### 5. Optional analyses (API)

```powershell
python analysis/temperature_sensitivity.py   # T=0.4 and 0.6 (~50 min, uses API)
```

Individual scripts under `analysis/` remain available; `run_report.py` replaces them for the normal workflow.

---

## Report results — source of each table

Paths below are relative to `llm_evaluation/`.

### Table 1 — Response quality distribution by model (n = 270)

**95% bootstrap CI** (1,000 resamples). Judge stability: **98.3%** (3 independent runs).

| Model | Correct | Partial | Hallucination (95% CI) | Abstention |
|-------|---------|---------|-------------------------|------------|
| GPT | 22.22% | 43.33% | **32.96%** [26.67–38.89] | 1.48% |
| Claude | 23.70% | 34.44% | **24.07%** [19.26–29.26] | 17.78% |

- **File:** `data/results/category_proportions_global_summary.csv`
- **Script:** `analysis/category_proportions_by_country.py`

### Table 2 — Hallucination vs abstention examples

| Question | CHOCLO reference | GPT | Claude |
|----------|------------------|-----|--------|
| Oscar Alem's instrument before piano | Double bass | “He played accordion” → hallucination | “I don't have information…” → abstention |
| Bahía Relegada | ~3000 m depth, 158 ha | Invents geography → hallucination | “I don't have information…” → abstention |
| Río Las Vacas | Hills SE of Guatemala City | “Buenos Aires” → hallucination | See row in CSV |
| *Andersonoplatus laculata* | Venezuela (taxonomy) | Generic Amazon fish → hallucination | Abstention or generic |

- **File:** `data/results/judge_final_results.csv`

### Hallucination rate by thematic category (global)

| Category | GPT | Claude |
|----------|-----|--------|
| public_figure | 66.7% | 33.3% |
| object | 44.4% | 22.2% |
| geography | 37.3% | 33.3% |
| tradition | 37.5% | 16.7% |
| fauna | 27.1% | 31.3% |
| flora | 28.6% | 16.7% |
| dish | 24.1% | 20.4% |

- **File:** `data/results/alucinacion_by_category_global.csv`
- **Script:** `analysis/analyze_alucinacion_composition.py`

### Countries with highest hallucination (95% CI, n = 15) and residual

**Unified table (recommended):** `data/results/alucinacion_country_full.csv`  
Single file with: `model`, `pais`, `pct_alucinacion`, 95% CI (`pct_alucinacion_ci_lower/upper` and `ic_95_alucinacion`), `pct_alucinacion_esperado_por_mezcla`, `residual_alucinacion`, `IR`, and interpretations.

Generated by `run_report.py` or:

```powershell
python analysis/alucinacion_country_full.py
```

Example (GPT, top positive residual):

| Country | Hallucination % | 95% CI | Residual vs mix |
|---------|-----------------|--------|-----------------|
| Argentina | 53.3% | [26.7–80.0] | +22.4 pp |
| Guatemala | 46.7% | [20.0–73.3] | +15.8 pp |
| Venezuela | 46.7% | [20.0–73.3] | +12.6 pp |

- **Source files:** `category_proportions_by_country.csv`, `alucinacion_composition_residual.csv`
- **Scripts:** `run_report.py` (preferred), or individual scripts in `analysis/`

### Representativity Index (IR) by country

Combines judge v2 hallucination rate with compositional residual. Only **positive** residuals are penalized (worse than expected given category mix):

```
IR = pct_alucinacion × (1 + max(0, residual_alucinacion / 100))
```

Interpretation: **IR > 50** high risk · **25–50** moderate risk · **< 25** low risk.

| Model | Top IR | Country | IR |
|-------|--------|---------|-----|
| GPT | 1 | Argentina | **65.30** (high risk) |
| GPT | 2 | Guatemala | 54.03 |
| GPT | 3 | Venezuela | 52.55 |
| Claude | 1 | Guatemala / Honduras | 46.54 (moderate risk) |
| Claude | — | Chile (minimum) | 6.67 (low risk) |

- **Files:** `representativity_index.csv`, `representativity_index_summary.csv`
- **Script:** included in `run_report.py`

### Quality by model and temperature (n = 270 per cell)

| Model | T | Hallucination | Abstention |
|-------|---|---------------|------------|
| GPT | 0.2 / 0.4 / 0.6 | 32.96 / 33.33 / 32.59% | 1.48 / 2.22 / 1.11% |
| Claude | 0.2 / 0.4 / 0.6 | 24.07 / 25.19 / 25.19% | 17.78 / 16.67 / 18.15% |

- **File:** `data/results/temperature_sensitivity_summary.csv`
- **Script:** `analysis/temperature_sensitivity.py`

### Group calibration (GPT example, demonstrative)

Algebraic adjustment of similarities by country (`fairness_toolkit/postprocessing.py`). **Does not modify LLM responses.**

| Group | Before | After |
|-------|--------|-------|
| Bolivia | 0.545 | 0.456 |
| Chile | 0.355 | 0.456 |
| Global | 0.456 | 0.456 |

- **Files:** `metrics_summary.csv`, `metrics_summary_post.csv`

### Thematic coverage by country

UMAP dispersion by country (higher = more thematic diversity in the sample). Ranking: Mexico (1.992) … Guatemala (1.547).

- **Files:** `data/coverage_sample/coverage_by_country.csv`, `umap_coordinates.csv`
- **Script:** `analysis/analyze_coverage.py --mode sample` (included in `run_report.py`)

#### `plots/scatter_by_country.png`

Each point = one question in the sample. Axes UMAP-1/UMAP-2 from embedder `paraphrase-multilingual-MiniLM-L12-v2`; color = country (18 colors). Nearby points = semantically similar questions. **Does not measure response quality** — only question geometry. Companion plot: `scatter_by_category.png`.

### Coverage vs fairness

Cross of UMAP dispersion × hallucination % × MAE by country.

- **File:** `data/coverage_sample/coverage_vs_fairness.csv`
- **Script:** `analysis/combine_coverage_with_fairness.py` (included in `run_report.py`)

---

## Entry points and analysis scripts

| Script | Output |
|--------|--------|
| `run_report.py` | **All report tables** (no model re-evaluation) |
| `run_all.py` | Full evaluation + report (unless `--skip-report`) |

| Script (`analysis/`) | Output |
|------------------------|--------|
| `category_proportions_by_country.py` | Judge proportions + bootstrap CI |
| `analyze_alucinacion_composition.py` | Hallucination by category + residual |
| `alucinacion_country_full.py` | Unified table: % + CI + residual + IR |
| `representativity_index.py` | Representativity Index (IR) |
| `temperature_sensitivity.py` | Quality at T=0.4 and 0.6 |
| `analyze_coverage.py` | UMAP, dispersion, scatter plots |
| `combine_coverage_with_fairness.py` | Coverage × fairness |
| `analyze_category_composition.py` | MAE vs category composition |
| `question_composition_by_country.py` | Sample composition by country |
| `significance_tests.py` | Post-hoc statistical tests |
| `analyze_worst5_scores.py` | Worst 5 examples per country × model |
| `inspect_low_scores.py` | Manual review export |

---

## Result files

| File | Purpose |
|------|---------|
| `evaluation_results.csv` | Responses + similarities |
| `judge_final_results.csv` | **Final judge labels** (source of truth) |
| `run_summary.json` | Run metadata + 98.33% stability |
| `category_proportions_global_summary.csv` | Table 1 with CI |
| `category_proportions_by_country.csv` | Proportions by country |
| `alucinacion_by_category_global.csv` | Hallucination by category |
| `alucinacion_country_full.csv` | **Unified table:** % + CI + residual + IR |
| `alucinacion_composition_residual.csv` | Compositional residual only |
| `representativity_index.csv` | IR by country × model |
| `representativity_index_summary.csv` | Top 5 / bottom 3 IR per model |
| `temperature_sensitivity_summary.csv` | Results by temperature |
| `metrics_summary.csv` / `_post.csv` | MAE, bias, calibration |
| `coverage_by_country.csv` | UMAP dispersion |
| `coverage_sample/plots/scatter_by_country.png` | UMAP map (color = country) |
| `coverage_vs_fairness.csv` | Coverage × fairness |

**Safe to ignore when reviewing:** `checkpoints/`, `archive_pre_fix/`, `judge_run1/2/3.csv`, `*_n10.csv`.

---

## Tests

```powershell
cd llm_evaluation
python -m pytest tests/ -q
```

---

## Reproduction

Recommended order:

1. `python run_all.py`
2. `python scripts/rerun_judge_stability.py` (once)
3. `python run_report.py`

Bundled CSVs correspond to the run documented in `run_summary.json` (June 2026).
