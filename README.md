# Crypto Narrative Hunter – Tesis de Máster

Este repositorio contiene los entregables completos para la tesis de máster de MBIT School **"Crypto Narrative Hunter: Clustering No Supervisado de Wallets Smart Money en Mercados DeFi de Ethereum."** Incluye únicamente los artefactos referenciados en el manuscrito final para que el jurado pueda auditar la investigación, reproducir el pipeline y revisar la documentación sin navegar por herramientas internas o archivos exploratorios.

## Mapa del Repositorio

| Ruta | Propósito |
| --- | --- |
| `thesis/` | Fuentes canónicas de la tesis (`.md` + `.html`) que coinciden con el manuscrito presentado. |
| `docs/product/` | Documentos de producto citados en las Secciones 4–7 (PRD, arquitectura, guía de implementación, estrategia MVP, plan Epic 4). |
| `docs/research/` | Anexos científicos: hipótesis de investigación, framework de evaluación, especificación de features. |
| `data-pipeline/` | Pipeline completo de datos de Ethereum (código, notebooks, SQL, tests, outputs curados). |
| `infra/` | `docker-compose.yml` para ejecutar los servicios de soporte descritos en la Sección 4. |
| `time-series-experiment/` | Experimento exploratorio con redes LSTM para predicción de precios de Ethereum (material complementario). |

> **Nota:** La carpeta `time-series-experiment/` contiene un pequeño ejercicio exploratorio de predicción de series temporales con redes neuronales LSTM aplicadas al precio de Ethereum. Aunque no forma parte del scope principal de la tesis, se incluye como material anecdótico que ilustra aproximaciones complementarias al análisis de mercados crypto.

## Descripción del Pipeline de Datos

La carpeta `data-pipeline/` es una instantánea directa y ejecutable del sistema de recolección de datos BMAD:

- **Runtime verificado**: Python **3.11.14** (build Homebrew). Los entornos 3.9 anteriores carecen de wheels compatibles para `contourpy`, `kiwisolver` y `hdbscan`, por lo que es necesario activar un intérprete 3.11 antes de sincronizar dependencias.
- **Paquete Python**: Ubicado en `data_collection/` con utilidades compartidas en `data_collection/common/` y módulos de servicio en `services/` (tokens, transacciones, balances, ingeniería de características, validación, etc.).
- **Puntos de entrada CLI**: Los scripts `cli_*.py` y de orquestación `run_*.py` reproducen todos los experimentos documentados en los Capítulos 5–6.
- **`analysis/`**: Agrupa el resumen ejecutivo de EDA, el informe completo de validación (12k+ palabras) y gráficos de soporte referenciados en el Apéndice A.
- **`notebooks/`**: Contiene los notebooks de Story 4.x citados en la tesis (EDA, clustering, interpretación, evaluación) más la presentación de Epic 4.
- **`outputs/`**: Recortado a los datasets, artefactos de clustering e informes mencionados explícitamente en la Sección 6 (archivos master/cleaned de wallet_features, métricas/visualizaciones de clustering, informe de limpieza y resúmenes de completitud de stories).
- **`sql/`**: Incluye los archivos de esquema y queries de Dune Analytics listadas en el Apéndice F.
- **`tests/`**: Contiene la cobertura unitaria/integración utilizada originalmente para validar el pipeline.

### Inicio Rápido

```bash
cd data-pipeline
uv venv
source .venv/bin/activate
uv pip sync pyproject.toml
cp .env.example .env   # editar con credenciales de Postgres + APIs
uv run data-collection init-db
uv run data-collection ensure-checkpoints
uv run python run_clustering_analysis.py
```

> **Nota:** Los cachés en bruto, entornos locales y assets exploratorios fueron intencionalmente excluidos para mantener el repositorio ligero. Todos los datasets necesarios para reproducir los resultados reportados están presentes en `outputs/`.

## Licencia y Atribución

Este repositorio se proporciona para evaluación académica. Por favor cita la tesis si reutilizas el material. Todas las fuentes de datos externas (Dune Analytics, Alchemy, CoinGecko) mantienen sus respectivas licencias y límites de tasa.

---

## Autores

**Oscar Pons, Antonio Nieves y Jose Luis Sanchez**

**Institución:** MBIT School
**Programa:** Máster en Inteligencia Artificial Avanzada y Generativa
**Fecha:** Noviembre 2025

---

<details>
<summary><b>English Version / Versión en Inglés</b></summary>

# Crypto Narrative Hunter – Thesis Release

This repository packages the deliverables required for the MBIT School master's thesis **"Crypto Narrative Hunter: Clustering No Supervisado de Wallets Smart Money en Mercados DeFi de Ethereum."** It contains only the artifacts referenced in the final manuscript so the jury can audit the research, reproduce the pipeline, and review the documentation without navigating internal tooling or exploratory leftovers.

## Repository Map

| Path | Purpose |
| --- | --- |
| `thesis/` | Canonical thesis sources (`.md` + `.html`) matching the submitted manuscript. |
| `docs/product/` | Product-facing documents cited in Sections 4–7 (PRD, architecture, implementation plan, MVP north star, Epic 4 plan). |
| `docs/research/` | Scientific annexes: research hypotheses, evaluation framework, feature specification. |
| `data-pipeline/` | Complete Ethereum data pipeline (code, notebooks, SQL, tests, curated outputs). |
| `infra/` | `docker-compose.yml` to run the supporting services described in Section 4. |
| `time-series-experiment/` | Exploratory LSTM neural network experiment for Ethereum price prediction (supplementary material). |

> **Note:** The `time-series-experiment/` folder contains a small exploratory exercise on time series prediction using LSTM neural networks applied to Ethereum prices. While not part of the thesis's main scope, it is included as anecdotal material illustrating complementary approaches to crypto market analysis.

## Data Pipeline Overview

The `data-pipeline/` folder is a direct, runnable snapshot of the BMAD data collection system:

- Verified runtime: Python **3.11.14** (Homebrew build). Earlier 3.9 environments lack compatible wheels for `contourpy`, `kiwisolver`, and `hdbscan`, so be sure to activate a 3.11 interpreter before syncing dependencies.
- Python package located under `data_collection/` with shared utilities in `data_collection/common/` and service modules inside `services/` (tokens, transactions, balances, feature engineering, validation, etc.).
- CLI entry points (`cli_*.py`) and orchestration scripts (`run_*.py`) reproduce every experiment documented in Chapters 5–6.
- `analysis/` groups the EDA executive summary, the full validation report (12k+ words), and supporting plots referenced in Appendix A.
- `notebooks/` contains the Story 4.x notebooks cited throughout the thesis (EDA, clustering, interpretation, evaluation) plus the Epic 4 presentation deck.
- `outputs/` is trimmed to the datasets, clustering artifacts, and reports explicitly mentioned in Section 6 (wallet_features master/cleaned files, clustering metrics/visualizations, cleanup report, and story completion summaries).
- `sql/` includes the schema files and Dune Analytics queries listed in Appendix F.
- `tests/` contains the unit/integration coverage originally used to validate the pipeline.

### Quick Start

```bash
cd data-pipeline
uv venv
source .venv/bin/activate
uv pip sync pyproject.toml
cp .env.example .env   # edit with Postgres + API secrets
uv run data-collection init-db
uv run data-collection ensure-checkpoints
uv run python run_clustering_analysis.py
```

> **Note:** The raw caches, local environments, and exploratory assets were intentionally excluded to keep the repository lean. All required datasets to reproduce the reported results are present under `outputs/`.

## License & Attribution

This repository is provided for academic evaluation. Please cite the thesis if you reuse the material. All external data sources (Dune Analytics, Alchemy, CoinGecko) retain their respective licenses and rate limits.

</details>
