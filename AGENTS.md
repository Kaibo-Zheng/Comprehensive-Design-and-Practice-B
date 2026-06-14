# Repository Guidelines

## Project Structure & Module Organization
This repository centers on the two-axis gimbal tracker. Keep real Python source at the repository root: `app.py`, `camera.py`, and subpackages such as `core/`, `inference/`, `motion/`, `web/`, and `voice/`. Keep C/C++ acceleration code under `acceleration/`. Keep tests under `tests/`. Demo wrappers live in `scripts/`, and deliverables in `result/`, `illustration/`, and `audio/`. `reports/` remains a compatibility link tree, and model files live under `model/`. Keep human-facing planning in `proposal.md` and task tracking in `tasks/` when those files are present. Treat `archive/` and `external/` as reference or vendored material; avoid editing them unless the task explicitly requires it. Models and large runtime artifacts belong under `model/` and should not be committed.

## Build, Test, and Development Commands
Use Python 3.10, preferably the documented conda env:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate b
```

Common commands:

```bash
python -m app --mock-servo --no-display --max-frames 5
python -m inference.tune_detector --detector face
python -m pytest tests -q
python -m build acceleration
bash scripts/run.sh
```

Use mock-servo or headless flags for routine validation. Run hardware scripts only on the target board with camera, PCA9685, and audio devices attached.

## Coding Style & Naming Conventions
Use 4-space indentation and keep runnable modules behind `if __name__ == "__main__":`. Prefer small, explicit functions and `argparse` options over hard-coded board paths. Use `snake_case` for files, functions, and variables; `PascalCase` for classes. Match existing module patterns such as detector backends in `inference/*detector*.py`.

## Testing Guidelines
Tests live in `tests/`. Add focused smoke or shape checks for reusable logic, especially post-processing, metrics, and voice helpers. Name tests `test_<feature>.py` and keep them CPU-safe. For hardware work, record the exact command, board setup, and log file written under `result/logs/`.

## Commit & Pull Request Guidelines
Local Git metadata may be absent, so use concise imperative commit subjects such as `Add RKNN detector smoke test` or `Fix web status serialization`. PRs should summarize changed modules, commands run, required hardware or model files, and attach screenshots or CSV/log references for UI, tracking, or benchmark changes.

## Security & Configuration Tips
Do not commit datasets, model weights, RKNN bundles, generated audio/video, virtual environments, or machine-specific absolute paths. Keep secrets and board-specific values in environment variables or CLI flags.

## Agent-Specific Instructions
For gimbal-tracker work, keep `proposal.md` as the human-facing proposal when present and implement code changes primarily in the root Python modules/packages, `acceleration/`, `tests/`, and `result/`. Only touch `archive/legacy_bonus/` when a task explicitly requires legacy bonus work. If `PROJECT_SPEC.md` is added later, read it before editing project code.
