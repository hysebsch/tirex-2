# TiRex-2 Task Tracker

> Rolling record of open and closed tasks. Updated by Claude at checkpoints to keep context compact.
> Format: `[OPEN|IN_PROGRESS|BLOCKED|CLOSED]` — subject — brief notes.

## Open / In Progress

- **[IN_PROGRESS]** Decide next TiRex Pro feature to implement
  - Core uv migration, training/fine-tuning hardening, and several Pro skeletons are implemented.
  - Streaming feature implemented as rolling-window forecaster.
  - Regression head implemented with frozen backbone + MLP head.
  - Remaining roadmap items: classification head, hardware compile/quantize backends.

## Pending / Needs Definition

- **[OPEN]** Open a pull request from `hysebsch/tirex-2:main` back to `NX-AI/tirex-2:main` when ready.
- **[OPEN]** Validate `make install-cuda` on DGX Spark with CUDA 13.0 driver (currently validated only on CPU / no GPU path).

## Closed

- **[CLOSED]** Remove Pixi from project.
- **[CLOSED]** Flatten repo so project root == Git root.
- **[CLOSED]** Add `Makefile` with install/test/lint/format/notebook targets.
- **[CLOSED]** Repository flattening and build-system migration (2026-08-03)
  - Moved from nested wrapper layout to flat Git root.
  - Replaced Pixi with `uv` + `Makefile` (`make install`, `make test`, `make lint`, etc.).
  - `pixi.lock` deleted; `.python-version` and `Makefile` added.
  - Files touched: `pyproject.toml`, `requirements.txt`, `.gitignore`, CI workflows.
- **[CLOSED]** Documentation refresh (2026-08-03)
  - `README.md`, `AGENT.md`, `examples/fevbench/README.md`, `examples/gifteval/README.md` modified.
  - HF token / weights access instructions removed.
  - DGX Spark / CUDA 12.8 notes added.
- **[CLOSED]** Pro/training code integration (2026-08-03)
  - New `src/tirex2/pro/` tree: training utilities, finetuning (`full`, `head-only`, `blocks`, `lora`), streaming, classification, regression, hardware skeletons.
  - New `test/test_training.py`.
  - `src/tirex2/model/tirex2.py` modified to preserve `_init_kwargs` for checkpoint reload.
- **[CLOSED]** CI/Docker updates (2026-08-03)
  - `.github/workflows/publish.yaml`, `.github/workflows/test.yaml` modified.
  - `inference/Dockerfile.cpu`, `inference/Dockerfile.gpu` modified.
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
- **[CLOSED]** Fork and push change set to `hysebsch/tirex-2` (2026-08-03)
  - Renamed local `origin` to `upstream` (NX-AI/tirex-2).
  - Added fork `https://github.com/hysebsch/tirex-2.git` as new `origin`.
  - Pushed two commits: uv/Pro training migration, and Pro skeleton hardening.
- **[CLOSED]** Harden Pro skeleton modules (2026-08-03)
  - `src/tirex2/pro/hardware/detect.py`: added `cuda_home` detection from `CUDA_HOME`/`CUDA_PATH` and system fallback; report in `print_hardware_report`.
  - `src/tirex2/pro/hardware/__init__.py`: export `HardwareInfo` and `print_hardware_report`.
  - Added `test/test_pro_skeletons.py` with 10 tests; `make test` now passes 101 tests.
- **[CLOSED]** Implement TiRex Pro streaming feature (2026-08-03)
  - Implemented `IncrementalForecaster` as a rolling-window streaming wrapper.
  - Accepts raw `TiRex2` or `ForecastModel`; wraps raw backbone in `ForecastModel` internally.
  - Supports target + past/future covariates, truncates to `context_length`.
  - Updated `test/test_pro_skeletons.py` with 4 streaming-specific tests.
  - `make test` passes 104 tests; `make lint` clean.
- **[CLOSED]** Implement TiRex Pro regression head (2026-08-03)
  - Added `TiRex2.forward_features()` to expose normalized stack output for downstream heads.
  - Implemented `TimeSeriesRegressor` with a frozen-backbone MLP head, mean-pooled target features, masked MSE loss.
  - Supports `fit`, `predict`, `save_head`, `load_head`; added 3 regression tests.
  - `make test` passes 105+ tests; `make lint` clean. One pre-existing flaky GPU-only flex-attention timing test occasionally fails.

---

## Checkpoint notes

- **2026-08-03 session start:** Context overfilled in prior session. User asked for regular summaries, compaction, and task tracking in `TASKS.md`.
- **2026-08-03 mid-session:** Training/fine-tuning hardening: fixed default config mismatch, added dataset validation, verified CLI smoke test and full `pytest test/` pass.
- **2026-08-03 end-session:** Packaging sanity and `make test` fixed. Fork created at `hysebsch/tirex-2`; both commits pushed. Pro skeletons hardened with tests. `make test` passes 101 tests; `make lint` clean.
- **2026-08-03 streaming session:** Implemented `IncrementalForecaster` rolling-window streaming feature. `make test` passes 104 tests; pushed to fork.
- **2026-08-03 regression session:** Implemented `TimeSeriesRegressor`. `make test` passes 105+ tests; pushed to fork. Note: pre-existing flaky GPU-only flex-attention timing test may fail intermittently.
- **Important constraints from `CLAUDE.md`:** CUDA 12.8 torch on NVIDIA; DGX Spark CUDA 13.0 driver is backward-compatible via local toolkit; do not commit `model/`, `output/`, `.venv/`, `__pycache__/`, `*.csv`, `.pixi/`, `*.egg-info`; use `uv`, not Pixi.
