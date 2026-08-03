# TiRex-2 Task Tracker

> Rolling record of open and closed tasks. Updated by Claude at checkpoints to keep context compact.
> Format: `[OPEN|IN_PROGRESS|BLOCKED|CLOSED]` — subject — brief notes.

## Open / In Progress

- **[IN_PROGRESS]** Repository flattening and build-system migration
  - Moved from nested wrapper layout to flat Git root.
  - Replaced Pixi with `uv` + `Makefile` (`make install`, `make test`, `make lint`, etc.).
  - `pixi.lock` deleted; `.python-version` and `Makefile` added.
  - Files touched: `pyproject.toml`, `requirements.txt`, `.gitignore`, CI workflows.

- **[IN_PROGRESS]** Documentation refresh
  - `README.md`, `AGENT.md`, `examples/fevbench/README.md`, `examples/gifteval/README.md` modified.
  - HF token / weights access instructions removed.
  - DGX Spark / CUDA 12.8 notes added.

- **[IN_PROGRESS]** Pro/training code integration
  - New `src/tirex2/pro/` tree: training utilities, finetuning (`full`, `head-only`, `blocks`, `lora`), streaming, classification, regression, hardware skeletons.
  - New `test/test_training.py`.
  - `src/tirex2/model/tirex2.py` modified.

- **[IN_PROGRESS]** CI/Docker updates
  - `.github/workflows/publish.yaml`, `.github/workflows/test.yaml` modified.
  - `inference/Dockerfile.cpu`, `inference/Dockerfile.gpu` modified.

## Pending / Needs Definition

- **[OPEN]** Review and commit the current large working-tree change set.
- **[OPEN]** Resolve any remaining import/issues introduced by the refactor.
- **[OPEN]** Decide whether Pro skeletons (streaming/classification/regression/hardware) need hardening before first release.

## Closed

- **[CLOSED]** Remove Pixi from project.
- **[CLOSED]** Flatten repo so project root == Git root.
- **[CLOSED]** Add `Makefile` with install/test/lint/format/notebook targets.
- **[CLOSED]** Re-establish context-compaction discipline (2026-08-03)
  - `TASKS.md` task tracker is the source of truth for open/closed work.
  - Claude will update this file after every significant subtask or ~3–5 active turns.
  - Claude will emit a compact checkpoint summary before context grows and ask to trim earlier detail.
- **[CLOSED]** Fix synthetic fine-tune config / window length mismatch (2026-08-03)
  - `configs/finetune.yaml`: `window_length` raised to 576 = 256 context + 320 prediction.
  - `FineTuner._ensure_dataset` now derives `context_length` from data length when not provided.
  - `TiRexDataset` raises a clear `ValueError` when zero windows are generated.
  - `scripts/train.py` passes `context_length`/`prediction_length` from config to `FineTuner.fit`.
  - CLI smoke test passed on CPU; full pytest suite passed (91 passed).
- **[CLOSED]** Packaging sanity check (2026-08-03)
  - `uv build` succeeded: `dist/tirex_2-0.2.1.tar.gz` and `dist/tirex_2-0.2.1-py3-none-any.whl`.
  - Wheel installs into a fresh venv and `from tirex2.pro.finetuning import FineTuner` works.
- **[CLOSED]** Run `make test` end-to-end (2026-08-03)
  - First `make test` failed because `.venv/bin` was not on PATH, so torch could not find the `ninja` binary.
  - Fixed by switching `test` / `test-single` targets to `uv run pytest`.
  - `make test` now passes: 91 passed, 16 warnings.
  - `make lint` still clean.

---

## Checkpoint notes

- **2026-08-03 session start:** Context overfilled in prior session. User asked for regular summaries, compaction, and task tracking in `TASKS.md`. This file updated; technical tasks above are carried over from the previous session and may need status verification before resuming work.
- **2026-08-03 mid-session:** Training/fine-tuning hardening: fixed default config mismatch, added dataset validation, verified CLI smoke test and full `pytest test/` pass. Lint clean.
- **2026-08-03 end-session:** Packaging sanity and `make test` fixed. Wheel builds, installs, imports. `make test` passes end-to-end. Remaining work: review/commit change set, decide on Pro skeleton hardening.
- **Important constraints from `CLAUDE.md`:** CUDA 12.8 torch on NVIDIA; DGX Spark CUDA 13.0 driver is backward-compatible via local toolkit; do not commit `model/`, `output/`, `.venv/`, `__pycache__/`, `*.csv`, `.pixi/`, `*.egg-info`; use `uv`, not Pixi.
