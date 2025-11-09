# Ethereum Data Collection Pipeline – Implementation Guide

_Last updated: 2025-09-27_

This guide explains how to set up, operate, and extend the data collection pipeline that underpins the Crypto Narrative Hunter thesis. It complements the high-level architecture document (`architecture.md`) by detailing concrete implementation choices, directory layout, tooling, and daily operational flows.

## 1. Solution Overview

- **Goal:** Deliver an end-to-end, restartable pipeline that assembles Ethereum token, wallet, transaction, balance, and pricing data, normalized to ETH and USD, with <1 GB storage footprint.
- **Scope (MVP):** Token metadata collection, smart wallet discovery, transaction extraction, balance snapshots, pricing backfill, and quality validation across Uniswap/Curve ecosystems.
- **Guiding Principles:** idempotent modules, low-cost operation on free-tier APIs, checkpoint-based resumability, and reproducibility suitable for academic validation.

## 2. Tech Stack Summary

| Layer | Selection | Rationale |
|-------|-----------|-----------|
| Language/runtime | Python 3.9+ | Mature ecosystem for data tooling, aligns with thesis requirements |
| Package/dependency management | [uv](https://github.com/astral-sh/uv) | Fast installer, deterministic sync, works with PEP 621 `pyproject.toml` |
| Database | PostgreSQL 15 | Relational staging, ACID semantics, robust tooling |
| Storage formats | Postgres + Parquet (snappy) | Postgres for staging, Parquet for analysis-friendly export |
| Logging | Python logging + JSON format | Machine-parseable logs ready for ingestion |
| Retry/backoff | `tenacity` | Declarative policies for exponential backoff |
| CLI | `argparse` + uv script entry point | Minimal friction for operators |

## 3. Repository Layout

```
data-collection/
  pyproject.toml          # uv-managed dependencies & CLI entry point
  README.md               # Quick start and command reference
  .env.example            # Sample configuration aligned with docker-compose
  sql/
    schema.sql            # Idempotent Postgres schema bootstrap
  data_collection/
    __init__.py
    cli.py                # Command dispatcher (init-db, health, checkpoint, ...)
    common/
      __init__.py
      config.py           # dotenv-backed settings loader
      logging_setup.py    # JSON logging to stdout + rotating file
      db.py               # Postgres connection helpers
      checkpoints.py      # Ensure + read/write checkpoint table
```

Additional directories such as `services/tokens/`, `services/wallets/`, etc., will be added as new stories are implemented.

## 4. Environment & Dependency Management

### 4.1 Using uv

1. **Create virtual environment**
   ```bash
   cd BMAD_TFM/data-collection
   uv venv
   source .venv/bin/activate
   ```

2. **Install dependencies** (sync to the exact versions/ranges defined in `pyproject.toml`):

   ### 4.3 Human-Owned Credential Workflow

   - **Owner:** Txelu Sánchez (project lead) provisions and refreshes all third-party API credentials (Dune, Alchemy, Etherscan, CoinGecko) and stores them in the team password vault.
   - **Handoff:** When new keys are issued, update `.env` locally and in any secure deployment secret store; notify developer agents via story notes.
   - **Rotation cadence:** Review quotas and rotate keys every 90 days or sooner if leaked/blocked; document changes in `docs/data-collection-phase/CHANGELOG.md` once created.
   - **Incident response:** If a credential is compromised, revoke it immediately in the provider dashboard, generate a new key, and log the incident in the project changelog.
   ```bash
   uv pip sync pyproject.toml
   ```

3. **Run commands** without manually activating the environment (optional):
   ```bash
   uv run data-collection health
   ```

4. **Add new dependencies**:
   ```bash
   uv pip install aiohttp
   uv pip compile pyproject.toml --all-extras  # optional lockfile generation
   ```

uv manages both the environment and dependency resolution; `requirements.txt` is retained solely for quick reference and backward compatibility.

### 4.2 Environment Variables

`data-collection/.env.example` documents all required variables. Copy it to `.env` and adjust as needed.

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | Connection string to Postgres. Default matches docker-compose (`postgresql://user:password@localhost:5432/crypto_narratives`). Use `database` hostname when running inside Docker network. |
| `LOG_LEVEL` | Logging verbosity (INFO, DEBUG, etc.). |
| `LOG_DIR` | Folder for rotating JSON log files. |
| API keys (`DUNE_API_KEY`, `ALCHEMY_API_KEY`, `ETHERSCAN_API_KEY`, `COINGECKO_API_KEY`) | Filled when corresponding integrations are implemented in later stories. |

## 5. Database Schema

Defined in `sql/schema.sql` and applied via `uv run data-collection init-db`.

- Tables mirror the PRD data model (`tokens`, `wallets`, `transactions`, `wallet_balances`, `eth_prices`, `collection_checkpoints`).
- Foreign keys enforce relationships (e.g., `transactions.wallet_address` → `wallets.wallet_address`).
- Indices speed common queries: `transactions` by wallet/block, `wallet_balances` by wallet/date.
- Schema is idempotent: rerunning `init-db` is safe.

Migration approach:

1. Modify `sql/schema.sql` with ALTER statements or create delta scripts.
2. Record the change in `docs/data-collection-phase/CHANGELOG.md` (to be created with future stories).
3. Re-run `uv run data-collection init-db`.

## 6. Logging Strategy

`common/logging_setup.py` configures:

- JSON output with keys: `level`, `message`, `logger`, `time`, plus optional `exc_info`.
- Console handler for immediate operator feedback.
- Rotating file handler (`collection.log`, 5 MB × 3 backups) stored in `LOG_DIR`.
- Called lazily by CLI commands to keep import-time side effects minimal.

## 7. Database Access Layer

### `common/db.py`

- Uses `psycopg2` for synchronous access.
- `get_cursor(readonly=False)` yields a context-managed cursor with auto-commit/rollback semantics.
- All operations log failures and re-raise for upstream handling.

### Future Enhancements

- Introduce connection pooling (`psycopg_pool`) once concurrency increases.
- Add async wrappers when migrating to `asyncpg` for async services.

## 8. Checkpoint Management

`common/checkpoints.py` exposes:

- `ensure_table()` — creates the checkpoint table if missing (also included in schema init).
- `get_checkpoint(type)` — returns the latest checkpoint row for a collection type (e.g., `tokens`, `wallets`, `transactions`).
- `upsert_checkpoint(...)` — current implementation appends rows (no update on conflict) to preserve audit history. Future stories can add a `last_updated` column or separate audit table as needed.

Usage pattern:

1. Before a collection run, call `checkpoint-show` to resume from previous state.
2. After each successful shard/batch, call `checkpoint-update` with new block height/date and status.
3. On failure, rerun service; logic will resume from last recorded checkpoint.

## 9. Command Line Interface

### Entry Points

- `uv run data-collection <command>` — preferred uv-managed execution.
- `python -m data_collection.cli <command>` — alternative when uv is unavailable.

### Available Commands (Story 1.1)

| Command | Description |
|---------|-------------|
| `init-db` | Reads `sql/schema.sql` and applies it idempotently. |
| `ensure-checkpoints` | Ensures the checkpoint table exists (redundant but safe). |
| `health` | Executes `SELECT 1` to confirm DB reachability. |
| `checkpoint-show --type <TYPE>` | Prints the latest checkpoint JSON for the given collection type. |
| `checkpoint-update --type <TYPE> [--block N] [--date YYYY-MM-DD] [--records M] [--status STATUS]` | Inserts a new checkpoint row with supplied metadata. |

### Example Session

```bash
uv run data-collection init-db
uv run data-collection ensure-checkpoints
uv run data-collection checkpoint-update --type tokens --status started --records 0
uv run data-collection checkpoint-show --type tokens
uv run data-collection health
```

The CLI will emit JSON logs to stdout and write rotating logs to the directory defined in `.env`.

## 10. Operational Workflows

### 10.1 Local Development Loop

1. Start Postgres (`docker compose up database`).
2. Activate uv environment (`uv venv && source .venv/bin/activate`).
3. Sync dependencies (`uv pip sync pyproject.toml`).
4. Copy `.env` and adjust DB host if running inside Docker network.
5. Initialize schema (`uv run data-collection init-db`).
6. Develop new service modules under `data_collection/services/...`.
7. Add tests (future story) and run them via `uv run pytest` (once configured).

### 10.2 Running Inside Docker Network

- Use `DATABASE_URL=postgresql://user:password@database:5432/crypto_narratives` in `.env`.
- Optionally create a dedicated container for the data collection service (future work) referencing the uv-managed project.

## 11. Extending the Pipeline (Roadmap)

| Story | Upcoming Deliverable | Key Files/Packages |
|-------|----------------------|--------------------|
| 1.2 | CoinGecko token ingestion | `services/tokens/collector.py`, `aiohttp`, caching layer |
| 1.3 | Dune liquidity analysis | Dune API wrapper, persistent cache, SQL templates |
| 2.x | Smart wallet queries & filtering | `services/wallets/`, heuristics module, MEV detection helper |
| 3.x | Transaction extraction & pricing | `services/transactions/`, `web3`, Chainlink pricing adapter, Multicall client |
| 4.x | Quality reporting | `reports/`, notebooks integration |

Each new service should:

1. Live under `data_collection/services/<name>/` with cohesive modules.
2. Expose CLI command(s) via `cli.py` dispatcher (e.g., `tokens collect`).
3. Respect checkpointing contract (read last checkpoint, process batch, update checkpoint).
4. Log structured metadata for observability.

## 12. Quality Assurance

### 12.1 Manual Validation (current scope)

- Run `uv run data-collection health` before/after schema updates.
- Inspect logs in `<LOG_DIR>/collection.log` for errors/exceptions.
- Verify tables in Postgres (`psql -h localhost -U user -d crypto_narratives -c '\dt'`).

### 12.2 Automated Validation (future stories)

- Add unit tests for helper modules (`pytest`).
- Implement data quality reports comparing counts vs. checkpoints.
- Include integration smoke tests to exercise service CLI commands against a seeded Postgres instance.

## 13. Troubleshooting

| Symptom | Likely Cause | Resolution |
|---------|--------------|------------|
| `psycopg2.OperationalError: could not connect to server` | Postgres container not running or wrong host in `DATABASE_URL`. | Start container (`docker compose up database`) or adjust `.env`. |
| `permission denied` when writing logs | `LOG_DIR` lacks write permission. | Change `LOG_DIR` or adjust file permissions. |
| `ON CONFLICT` errors when updating checkpoints | The current implementation inserts rows without updating existing ones. Remove conflicting rows or adjust `collection_checkpoints` manually. Future enhancement will support true UPSERT semantics. |

## 14. Documentation Map

- `docs/data-collection-phase/architecture.md` — system blueprint & constraints.
- `docs/data-collection-phase/implementation-guide.md` (this file) — how-to, operations, roadmap.
- `data-collection/README.md` — quick reference for developers.
- `docs/data-collection-phase/prd.md` — functional/non-functional requirements.

Maintain these documents in tandem as features evolve. Significant changes must note impact in `architecture.md` and reference the implementation guide for actionable steps.

## 15. Contact & Ownership

- **Engineering Lead:** Winston (Architect persona) — architecture integrity, technology decisions.
- **Implementation Lead:** Development agent(s) — deliver stories following this playbook.
- **QA Lead:** Quinn — quality gates, risk assessments.

Updates to architecture or core assumptions should route through the Architect persona to keep documentation and implementation aligned.

---

For questions or clarifications, open an issue in the repo or tag the appropriate persona via BMAD commands. Continue logging lessons learned in the forthcoming `CHANGELOG.md` to preserve institutional knowledge.

## 16. CI & Testing Plan

### 16.1 Objectives

- Guarantee every change runs linting, unit tests, and data-validation smoke checks before merge.
- Keep runtimes under 8 minutes so the 13-day delivery window is not impacted.
- Align local developer workflows with the automated pipeline to avoid surprises.

### 16.2 Local Workflow

All commands execute from `BMAD_TFM/data-collection/`.

```bash
# Install/update environment
uv venv
source .venv/bin/activate
uv pip sync pyproject.toml

# Static analysis (ruff or flake8 once configured)
uv run ruff check .

# Formatting check (optional but recommended)
uv run ruff format --check .

# Unit test entry point (pytest harness added in Story 1.1)
uv run pytest

# Data-validation smoke (invoked after implementing Story 1.2+)
uv run data-collection validate --sample-size 50
```

### 16.3 CI Pipeline (GitHub Actions Reference)

1. **Trigger:** `push` and `pull_request` against `main` and release branches.
2. **Setup:**
   - Use `actions/setup-python@v5` with Python 3.11 (compatible with 3.9+ code).
   - Cache the uv virtual environment (`~/.cache/uv`) to accelerate installs.
3. **Steps:**
   1. `uv pip sync pyproject.toml`
   2. `uv run ruff check .`
   3. `uv run pytest`
   4. `uv run data-collection validate --sample-size 20 --dry-run` (skipped until command exists; keep as TODO)
   5. Upload test results (`pytest.xml`) and coverage to the Actions summary.
4. **Artifacts:** Publish the latest `sql/schema.sql` and generated QA reports for traceability when the validation command is active.

> **Note:** The validation command will act as a harness that reads the latest Parquet outputs or sample batches, ensuring schema compliance without running the full pipeline. Implement the CLI stub during Story 1.1 and flesh it out alongside Story 1.2/1.3 deliverables.

### 16.4 Testing Strategy Roadmap

| Stage | Test Level | Owner | Notes |
|-------|------------|-------|-------|
| Story 1.1 | Static analysis + unit tests for `common/` modules | Dev Agent | Establish pytest project, add smoke tests for DB + checkpoint utilities. |
| Story 1.2 | Integration tests for CoinGecko collector (mocked APIs) | Dev Agent | Use `responses`/`pytest-asyncio` to simulate rate limits; assert caching logic. |
| Story 1.3 | Contract tests for Dune SQL output schema | Dev Agent | Snapshot expected columns; ensure TVL calculations match sample results. |
| Epic 2 | Data-quality checks (completeness, filters) | QA Agent | Generate QA reports via CLI, fail the pipeline when coverage <95%. |
| Epic 3 | End-to-end replay on 10 wallet sample | QA Agent | Ensure swap decoding + balance reconciliation stays deterministic. |

### 16.5 Failing the Build

- Lint/test/validation failures block merges; fixes must be pushed before PR approval.
- Re-run workflows after updating secrets or environment by using the “Re-run jobs” action to confirm stability.
- Document flaky tests in `/docs/data-collection-phase/CHANGELOG.md` with resolution steps.
