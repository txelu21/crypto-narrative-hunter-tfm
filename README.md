# Crypto Narrative Hunter – Thesis Release

This repository packages the deliverables required for the MBIT School master's thesis **“Crypto Narrative Hunter: Clustering No Supervisado de Wallets Smart Money en Mercados DeFi de Ethereum.”** It contains only the artifacts referenced in the final manuscript so the jury can audit the research, reproduce the pipeline, and review the documentation without navigating internal tooling or exploratory leftovers.

## Repository Map

| Path | Purpose |
| --- | --- |
| `thesis/` | Canonical thesis sources (`.md` + `.html`) matching the submitted manuscript. |
| `docs/product/` | Product-facing documents cited in Sections 4–7 (PRD, architecture, implementation plan, MVP north star, Epic 4 plan). |
| `docs/research/` | Scientific annexes: research hypotheses, evaluation framework, feature specification. |
| `data-pipeline/` | Complete Ethereum data pipeline (code, notebooks, SQL, tests, curated outputs). |
| `infra/` | `docker-compose.yml` to run the supporting services described in Section 4. |

## Data Pipeline Overview

The `data-pipeline/` folder is a direct, runnable snapshot of the BMAD data collection system:

- Verified runtime: Python **3.11.14** (Homebrew build). Earlier 3.9 environments lack compatible wheels for `contourpy`, `kiwisolver`, and `hdbscan`, so be sure to activate a 3.11 interpreter before syncing dependencies.
- Python package located under `data_collection/` with shared utilities in `data_collection/common/` and service modules inside `services/` (tokens, transactions, balances, feature engineering, validation, etc.).
- CLI entry points (`cli_*.py`) and orchestration scripts (`run_*.py`) reproduce every experiment documented in Chapters 5–6.
- `analysis/` groups the EDA executive summary, the full validation report (12k+ words), and supporting plots referenced in Appendix A.
- `notebooks/` contains the Story 4.x notebooks cited throughout the thesis (EDA, clustering, interpretation, evaluation) plus the Epic 4 presentation deck.
- `outputs/` is trimmed to the datasets, clustering artifacts, and reports explicitly mentioned in Section 6 (wallet_features master/cleaned files, clustering metrics/visualizations, cleanup report, and story completion summaries).
- `sql/` includes the schema files and Dune Analytics queries listed in Appendix F.
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
