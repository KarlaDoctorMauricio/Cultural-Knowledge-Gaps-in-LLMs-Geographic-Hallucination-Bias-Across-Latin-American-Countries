# Evaluación de fairness en LLMs sobre CHOCLO

Este repositorio contiene el código y los resultados de una evaluación comparativa de **GPT-4o-mini** (`gpt-4o-mini`) y **Claude Haiku 4.5** (`claude-haiku-4-5-20251001`) sobre el benchmark cultural latinoamericano [CHOCLO](https://huggingface.co/datasets/latam-gpt/CHOCLO).

Los IDs de modelo se configuran en `llm_evaluation/clients.py` (o vía `OPENAI_MODEL` / `ANTHROPIC_MODEL` en `.env`). El **judge** de calidad usa los mismos modelos: GPT como clasificador principal y Claude como respaldo.

Se midió:

1. **Similitud semántica** entre respuesta del modelo y referencia CHOCLO (embeddings multilingües).
2. **Calidad de respuesta** con un judge LLM en cuatro categorías: correcta, parcial, alucinación, abstención.
3. **Fairness por país** (MAE, sesgo, proporciones de alucinación con intervalos de confianza).
4. **Cobertura temática** del sample (dispersión en espacio UMAP de embeddings).
5. **Índice de Representatividad (IR)** por país, combinando alucinación observada y residual composicional.

**Muestra evaluada:** 270 preguntas, 15 por país, **18 países** (Brasil no aparece en esta muestra). Temperatura de generación **T = 0,2**. Los análisis estadísticos (muestreo, bootstrap, UMAP) usan semilla fija **42**; las llamadas a la API no usan `seed`.

---

## Qué hay en este repositorio

```
F-KADM/
├── README.md                 ← este documento (única guía del proyecto)
├── .env.example              ← plantilla de API keys (copiar a .env)
└── llm_evaluation/
    ├── run_all.py            ← evaluación + tablas del informe (paso 6)
    ├── run_report.py         ← solo tablas del informe (sin re-evaluar)
    ├── evaluator.py          ← judge LLM
    ├── clients.py            ← API OpenAI / Anthropic
    ├── fairness_toolkit/     ← pipeline (embeddings, MAE, calibración)
    ├── analysis/             ← scripts post-hoc (tablas del informe)
    ├── scripts/              ← muestreo y estabilidad del judge
    ├── tests/
    └── data/
        ├── choclo_sample.csv
        ├── results/          ← CSVs de resultados
        └── coverage_sample/  ← UMAP, cobertura y gráficos
```

Todo el código ejecutable está en `llm_evaluation/`.

---

## Cómo instalar y ejecutar

### 1. Dependencias

```powershell
cd llm_evaluation
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. API keys

En la raíz del repo, copia `.env.example` a `.env`:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

### 3. Evaluación + tablas del informe (un comando)

```powershell
python run_all.py
```

Genera respuestas, similitudes, métricas de fairness, judge, **y al final** las tablas del informe (IR, `alucinacion_country_full.csv`, UMAP). Reanuda desde checkpoint si se interrumpe.

Opciones útiles:

```powershell
python run_all.py --skip-report          # solo evaluación, sin UMAP/IR
python run_all.py --skip-coverage        # report sin UMAP (más rápido)
python run_all.py -n 30                  # prueba rápida
```

### 4. Judge consolidado (estabilidad 98,3 %) — una vez

Tras fijar el prompt del judge, ejecutar **una vez** antes del reporte definitivo:

```powershell
python scripts/rerun_judge_stability.py
```

Salida canónica: `judge_final_results.csv` (moda de 3 corridas). Luego:

```powershell
python run_report.py
```

`run_report.py` regenera **todas** las tablas del informe desde `judge_final_results.csv` (proporciones, residual, IR, tabla unificada, UMAP). No llama a GPT/Claude para respuestas.

```powershell
python run_report.py --skip-coverage     # sin UMAP (~1 min)
```

### 5. Análisis opcionales (API)

```powershell
python analysis/temperature_sensitivity.py   # T=0.4 y 0.6 (~50 min, usa API)
```

Los scripts individuales en `analysis/` siguen disponibles; `run_report.py` los sustituye para el flujo normal.

---

## Resultados del informe — origen de cada tabla

### Tabla 1 — Distribución de calidad por modelo (n = 270)

IC **95 % bootstrap** (1.000 remuestreos). Estabilidad del judge: **98,3 %**.

| Modelo | Correcta | Parcial | Alucinación (IC 95 %) | Abstención |
|--------|----------|---------|------------------------|------------|
| GPT | 22,22 % | 43,33 % | **32,96 %** [26,67–38,89] | 1,48 % |
| Claude | 23,70 % | 34,44 % | **24,07 %** [19,26–29,26] | 17,78 % |

- **Archivo:** `data/results/category_proportions_global_summary.csv`
- **Script:** `analysis/category_proportions_by_country.py`

### Tabla 2 — Ejemplos alucinación vs abstención

| Pregunta | Referencia CHOCLO | GPT | Claude |
|----------|-------------------|-----|--------|
| Instrumento de Oscar Alem antes del piano | Contrabajo | «Tocó el acordeón» → alucinación | «No tengo información…» → abstención |
| Bahía Relegada | ~3000 m, 158 ha | Inventa geografía → alucinación | «No tengo información…» → abstención |
| Río Las Vacas | Cerros SE de Ciudad de Guatemala | «Buenos Aires» → alucinación | Ver fila en CSV |
| *Andersonoplatus laculata* | Venezuela (taxonomía) | Pez amazónico genérico → alucinación | Abstención o genérico |

- **Archivo:** `data/results/judge_final_results.csv`

### Alucinación por categoría temática (global)

| Categoría | GPT | Claude |
|-----------|-----|--------|
| public_figure | 66,7 % | 33,3 % |
| object | 44,4 % | 22,2 % |
| geography | 37,3 % | 33,3 % |
| tradition | 37,5 % | 16,7 % |
| fauna | 27,1 % | 31,3 % |
| flora | 28,6 % | 16,7 % |
| dish | 24,1 % | 20,4 % |

- **Archivo:** `data/results/alucinacion_by_category_global.csv`
- **Script:** `analysis/analyze_alucinacion_composition.py`

### Países con mayor alucinación (IC 95 %, n = 15) y residual

**Tabla unificada (recomendada):** `data/results/alucinacion_country_full.csv`  
Une en un solo archivo: `model`, `pais`, `pct_alucinacion`, IC 95 % (`pct_alucinacion_ci_lower/upper` e `ic_95_alucinacion`), `pct_alucinacion_esperado_por_mezcla`, `residual_alucinacion`, `IR` e interpretaciones.

```powershell
python analysis/alucinacion_country_full.py
```

Ejemplo (GPT, top residual positivo):

| País | % alucinación | IC 95 % | Residual vs. mezcla |
|------|---------------|---------|---------------------|
| Argentina | 53,3 % | [26,7–80,0] | +22,4 pp |
| Guatemala | 46,7 % | [20,0–73,3] | +15,8 pp |
| Venezuela | 46,7 % | [20,0–73,3] | +12,6 pp |

- **Archivos fuente:** `category_proportions_by_country.csv`, `alucinacion_composition_residual.csv`
- **Scripts:** `analysis/category_proportions_by_country.py`, `analysis/analyze_alucinacion_composition.py`, `analysis/alucinacion_country_full.py`

### Índice de Representatividad (IR) por país

Combina la tasa de alucinación del judge v2 con el residual composicional. Solo penaliza residuales **positivos** (peor de lo esperado por la mezcla de categorías):

```
IR = pct_alucinacion × (1 + max(0, residual_alucinacion / 100))
```

Interpretación: **IR > 50** riesgo alto · **25–50** riesgo moderado · **< 25** riesgo bajo.

| Modelo | Top IR | País | IR |
|--------|--------|------|-----|
| GPT | 1 | Argentina | **65,30** (riesgo alto) |
| GPT | 2 | Guatemala | 54,03 |
| GPT | 3 | Venezuela | 52,55 |
| Claude | 1 | Guatemala / Honduras | 46,54 (riesgo moderado) |
| Claude | — | Chile (mínimo) | 6,67 (riesgo bajo) |

- **Archivos:** `representativity_index.csv`, `representativity_index_summary.csv`
- **Script:** `analysis/representativity_index.py` (lee `alucinacion_composition_residual.csv`)

### Calidad por modelo y temperatura (n = 270 por celda)

| Modelo | T | Alucinación | Abstención |
|--------|---|-------------|------------|
| GPT | 0,2 / 0,4 / 0,6 | 32,96 / 33,33 / 32,59 % | 1,48 / 2,22 / 1,11 % |
| Claude | 0,2 / 0,4 / 0,6 | 24,07 / 25,19 / 25,19 % | 17,78 / 16,67 / 18,15 % |

- **Archivo:** `data/results/temperature_sensitivity_summary.csv`
- **Script:** `analysis/temperature_sensitivity.py`

### Calibración por grupo (ejemplo GPT, demostrativo)

Ajuste algebraico de similitudes por país (`fairness_toolkit/postprocessing.py`). **No modifica respuestas del LLM.**

| Grupo | Antes | Después |
|-------|-------|---------|
| Bolivia | 0,545 | 0,456 |
| Chile | 0,355 | 0,456 |
| Global | 0,456 | 0,456 |

- **Archivos:** `metrics_summary.csv`, `metrics_summary_post.csv`

### Cobertura temática por país

Dispersión UMAP por país (mayor = más diversidad temática en el sample). Ranking: México (1,992) … Guatemala (1,547).

- **Archivos:** `data/coverage_sample/coverage_by_country.csv`, `umap_coordinates.csv`
- **Script:** `analysis/analyze_coverage.py --mode sample`

#### `plots/scatter_by_country.png`

Cada punto = una pregunta del sample. Ejes UMAP-1/UMAP-2 tras embedder `paraphrase-multilingual-MiniLM-L12-v2`; color = país (18 colores). Puntos cercanos = preguntas semánticamente similares. **No mide calidad de respuestas**, solo geometría de las preguntas. Hermana: `scatter_by_category.png`.

### Cobertura vs fairness

Cruce dispersión UMAP × % alucinación × MAE por país.

- **Archivo:** `data/coverage_sample/coverage_vs_fairness.csv`
- **Script:** `analysis/combine_coverage_with_fairness.py`

---

## Scripts de análisis (`analysis/`)

| Script | Genera |
|--------|--------|
| `run_report.py` | **Todas las tablas del informe** (sin re-evaluar modelos) |
| `category_proportions_by_country.py` | Proporciones judge + IC bootstrap |
| `analyze_alucinacion_composition.py` | Alucinación por categoría y residual |
| `alucinacion_country_full.py` | Tabla unificada % + IC + residual + IR |
| `representativity_index.py` | Índice de Representatividad (IR) |
| `temperature_sensitivity.py` | Calidad a T=0.4 y 0.6 |
| `analyze_coverage.py` | UMAP, dispersión, scatter plots |
| `combine_coverage_with_fairness.py` | Cobertura × fairness |
| `analyze_category_composition.py` | MAE vs composición de categorías |
| `question_composition_by_country.py` | Composición del sample por país |
| `significance_tests.py` | Tests estadísticos post-hoc |
| `analyze_worst5_scores.py` | Peores 5 ejemplos por país × modelo |
| `inspect_low_scores.py` | Export revisión manual |

---

## Archivos de resultados

| Archivo | Uso |
|---------|-----|
| `evaluation_results.csv` | Respuestas + similitudes |
| `judge_final_results.csv` | **Etiquetas finales del judge** |
| `run_summary.json` | Metadatos + estabilidad 98,33 % |
| `category_proportions_global_summary.csv` | Tabla 1 con IC |
| `category_proportions_by_country.csv` | Proporciones por país |
| `alucinacion_by_category_global.csv` | Alucinación por categoría |
| `alucinacion_country_full.csv` | **Tabla unificada:** % + IC + residual + IR por país |
| `alucinacion_composition_residual.csv` | Residual composicional (solo) |
| `representativity_index.csv` | IR por país × modelo |
| `representativity_index_summary.csv` | Top 5 / bottom 3 IR por modelo |
| `temperature_sensitivity_summary.csv` | Resultados por temperatura |
| `metrics_summary.csv` / `_post.csv` | MAE, bias, calibración |
| `coverage_by_country.csv` | Dispersión UMAP |
| `coverage_sample/plots/scatter_by_country.png` | Mapa UMAP (color = país) |
| `coverage_vs_fairness.csv` | Cobertura × fairness |

**Ignorar al revisar:** `checkpoints/`, `archive_pre_fix/`, `judge_run1/2/3.csv`, `*_n10.csv`.

---

## Tests

```powershell
cd llm_evaluation
python -m pytest tests/ -q
```

---

## Reproducción

Ejecutar pasos 3–5 en orden. Los CSVs incluidos corresponden a la corrida en `run_summary.json` (junio 2026).
