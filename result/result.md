# Gimbal Person Tracker Result Notes

## Current Status

Implemented the first working version of the two-axis gimbal person tracker:

- `app.py`: camera + person detection + controller + gimbal + metrics.
- `inference/detector.py`: OpenCV face detector by default, with `upperbody` and `hog` options.
- `motion/gimbal.py`: PCA9685 servo wrapper with conservative angle limits and mock mode.
- `inference/tune_detector.py`: preview detection without moving the gimbal.
- `acceleration/`: Python and C extension 5x5 convolution benchmark.

## Run Commands

Use the conda environment `b`, which includes OpenCV, PyTorch CPU, smbus2, and the C extension package:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate b
```

Mock servo smoke test:

```bash
python -m app --mock-servo --no-display --max-frames 5 --log reports/logs/tracking_smoke.csv
```

Detector preview:

```bash
python -m inference.tune_detector --detector face
```

Detector preview with a stable USB camera path:

```bash
python -m inference.tune_detector --camera /dev/v4l/by-id/<ugreen-video-index0> --detector face
```

Real gimbal run:

```bash
python -m app --detector face --log reports/logs/tracking_log.csv
```

Real gimbal run with UGREEN or another selected USB camera:

```bash
python -m app --camera /dev/v4l/by-id/<ugreen-video-index0> --detector face --log reports/logs/tracking_log.csv
```

If one axis moves in the wrong direction, flip the sign:

```bash
python -m app --pan-sign -1
python -m app --tilt-sign 1
```

Opt-in safe calibration:

```bash
python -m app --calibrate
```

Mock calibration check:

```bash
python -m app --mock-servo --calibrate
```

Observed path:

```text
pan: 95 -> 85 -> 90
tilt: 95 -> 85 -> 90
```

## Smoke Test Results

Command:

```bash
python -m app --mock-servo --no-display --max-frames 5 --log reports/logs/tracking_smoke.csv
```

Observed output:

```text
frames: 5
runtime_s: 2.062
average_fps: 2.425
ema_fps: 8.477
average_error_px: 221.547
max_error_px: 222.657
lost_frames: 3
```

This confirms that camera capture, OpenCV detection, mock-servo control, and CSV logging run together.

## 5x5 Convolution Bonus

Install build tools if needed:

```bash
python -m pip install build wheel
```

Build C extension in place:

```bash
python acceleration/setup.py build_ext --inplace
```

Build wheel:

```bash
python -m build acceleration
```

Built artifacts:

```text
acceleration/dist/cpp_conv-0.1.0.tar.gz
acceleration/dist/cpp_conv-0.1.0-cp310-cp310-linux_aarch64.whl
```

Wheel install and import validation:

```bash
python -m pip install --force-reinstall acceleration/dist/cpp_conv-0.1.0-cp310-cp310-linux_aarch64.whl
cd /tmp
/root/miniconda3/envs/b/bin/python -c "import cpp_conv; print(cpp_conv.__file__)"
```

Observed installed import path:

```text
/root/miniconda3/envs/b/lib/python3.10/site-packages/cpp_conv/__init__.py
```

Benchmark command:

```bash
python acceleration/benchmark.py --height 120 --width 160 --iterations 3
```

Observed output:

```text
Frame size: 160x120
Iterations: 3
Max output diff: 0.000046
Python 5x5 convolution: 484.398 ms/frame
C extension 5x5 convolution: 0.889 ms/frame
Speedup: 545.13x
```

## Remaining Hardware Validation

The camera and PCA9685 were previously readable from the board. The next manual step is to run safe calibration and confirm:

- `pan_channel=0` controls the horizontal axis.
- `tilt_channel=1` controls the vertical axis.
- `pan_range=45..135` and `tilt_range=60..120` do not hit mechanical limits.
- `--pan-sign` and `--tilt-sign` values move the camera toward the person rather than away.

---

# Multi-Agent Task Execution (proposal.md protocol)

## TASK-000 baseline smoke test — done (2026-06-10)

Camera `/dev/video0` is a free HD 720P UVC webcam (`fuser` shows no occupant).

Command:

```bash
/root/miniconda3/envs/b/bin/python3 -m src.app \
  --camera /dev/video0 --detector face --mock-servo --no-display \
  --max-frames 30 --log reports/logs/p0_mock_smoke.csv
```

Summary:

```text
frames: 30
average_fps: 9.31
ema_fps: 22.07
average_error_px: 0.0
max_error_px: 0.0
lost_frames: 30   # no person in headless view, expected
```

CSV `reports/logs/p0_mock_smoke.csv` created with header
`frame,timestamp_s,fps,found,stale,x_error,y_error,error_px,pan,tilt` and 30 data rows.
Memory: available 2.1Gi before/after.

## TASK-001 project structure — done (2026-06-10)

- New `camera.py` holds `open_camera()` (moved from `app.py`); `app.py` imports it.
- New `model/README.md` (no-commit rule + ONNX/RKNN pipeline), `scripts/README.md`, `reports/screenshots/.gitkeep`.
- Validation: `python -m compileall src` OK; 5-frame mock run unchanged.

## TASK-009 cpp_conv benchmark — done (2026-06-10)

```text
Frame size: 160x120, iterations 3
Max output diff: 0.000046
Python 5x5 convolution: 751.198 ms/frame
C extension 5x5 convolution: 0.895 ms/frame
Speedup: 839.77x
```

- Saved to `reports/logs/cpp_conv_benchmark.txt`.
- Wheel `cpp_conv-0.1.0-cp310-cp310-linux_aarch64.whl` already installed; imports and runs `conv5x5` from `/tmp` (non-source dir).
- `pymp` not present in env `b`; optional parallel comparison recorded as untestable.

## TASK-002 postprocess + tests — done (2026-06-10)

- New `inference/postprocess.py`: box conversions, `clip_box`, `scale_box`, `iou`/`iou_batch`, greedy `nms`, `letterbox_params`, `scale_coords_letterbox`, `select_target`. Framework-independent (numpy only).
- New `tests/test_postprocess.py`: 7 tests covering overlap/disjoint/contained IoU, NMS suppression, no-negative-size clipping, letterbox round-trip, class-filtered target selection.
- pytest not in env `b`; ran `python tests/test_postprocess.py` -> 7/7 passed. (Root tests/ is Ultralytics'; the file runs standalone.)

## TASK-004 metrics schema upgrade — done (2026-06-10)

- `metrics.py` CSV header is now:
  `timestamp,frame_index,fps,detector,target_found,target_stale,target_label,target_confidence,x_error,y_error,error_norm,pan,tilt,capture_ms,preprocess_ms,inference_ms,postprocess_ms,total_ms,lost_frames`
- `summary()` adds `detector`, `avg_capture_ms`, `avg_inference_ms`, `avg_total_ms`.
- `app.py` times capture + detect per frame; detectors may expose a `last_timing` dict for a preprocess/inference/postprocess breakdown. Added `Target.confidence`.
- Validation: 10-frame mock run produced `reports/logs/p4_metrics_schema.csv` with the full header; old smoke command path unchanged.

## TASK-003 web monitor integration — done (2026-06-10)

- New `web/monitor.py`: stdlib `ThreadingHTTPServer`, thread-safe `WebState`, `WebMonitor` lifecycle. Endpoints `/`, `/stream.mjpg`, `/snapshot.jpg`, `/status.json` (+`do_HEAD`).
- `app.py`: `--web-host`, `--web-port`, `--web-quality`. Each frame the loop JPEG-encodes the overlay and pushes frame+status; on exit `monitor.stop()` shuts the server and frees the port.
- Validation (`--web-host 127.0.0.1 --web-port 8080 --no-display`): status.json returned full JSON; `curl -I snapshot.jpg` -> 200 image/jpeg; GET snapshot -> 640x480 JPEG (saved `reports/screenshots/web_snapshot.jpg`); stream.mjpg emitted multipart frames; SIGINT stopped process and freed 8080.
- Note: omit the GUI window with `--no-display` on the headless board (Qt xcb needs X).

## TASK-006 RKNN env + conversion — done (2026-06-10; refreshed 2026-06-13)

Board RKNN/NPU status (`reports/logs/rknn_env.txt`):

```text
NPU device-tree status : okay
RKNPU driver           : v0.9.6
/usr/lib/librknnrt.so  : present (2023-05-23)
/usr/bin/rknn_server   : present
/dev/dri/renderD128/9  : present
Python rknnlite        : OK via rknnlite.api (env b, refreshed 2026-06-13)
onnxruntime            : 1.23.2 (CPUExecutionProvider)
```

- `scripts/check_rknn_env.sh` — read-only collector (modules, libs, render nodes, dmesg, driver version).
- `scripts/convert_yolo_rknn.sh` / `tool/convert_yolo_rknn.py` — ONNX->RKNN via `rknn.api.RKNN`; lazy import + clear error when toolkit missing (exit 3) or ONNX missing (exit 2). Conversion is an x86-host step.
- Original blocker was missing `rknnlite`; this is now resolved. The refreshed `reports/logs/rknn_env.txt` shows `rknnlite.api` OK and `librknnrt.so` present.

## TASK-005 YOLO ONNX CPU detector — done (2026-06-11)

This was the task interrupted mid-edit. Root cause: `app.py` still called `PersonDetector(detector_config)` directly; the `build_detector()` factory (face/upperbody/hog + lazy `yolo_onnx`/`yolo_rknn`), the CLI args, the `DetectorConfig` fields, the `YoloOnnxDetector`, and the `preprocess_yolo`/`decode_yolo_output`/`select_target` helpers were all already present — only the one call site was never switched. Fixed `app.py:174` to `build_detector(detector_config)` and removed a stale atomic-write temp `app.py.tmp.31925.1be413ce239a` (which contained exactly that fix).

Validation:

- A. OpenCV unchanged: `--detector face --mock-servo --no-display --max-frames 5` -> clean JSON summary, `detector: face`.
- B. Missing model: `--detector yolo_onnx --model model/missing.onnx` (absent) -> `error: ONNX model not found: ...` and exit 1, no traceback.
- C. Real inference: `model/yolo11n.onnx` is the local COCO ONNX model; `person` is tracked via `--target-class person`. 5-frame run, cgroup-capped:

```text
detector=yolo_onnx frames=5
avg_capture_ms   ~251   (UVC read, dominates FPS)
avg_inference_ms ~137
preprocess_ms    ~7-9
postprocess_ms   ~2-3
avg_total_ms     ~405
average_fps      ~2.0
lost_frames      5   (headless, no person in view)
```

`decode_yolo_output` handled yolo11n output `(1,84,8400)` via the v8/v11 branch (84 = 4 box + 80 classes). CSV `reports/logs/p5_yolo_onnx.csv` has the full TASK-004 schema incl. per-stage timing.

Memory check (TASK-005):

```text
- The earlier crash was the torch+ultralytics IMPORT probe (heavy on 4GB), NOT ONNX inference.
  The yolo_onnx detector imports only onnxruntime.
- Before: available 1.8Gi. After: available 1.8Gi, swap 0B.
- Top memory processes: vscode-server node (~600MB), pylance (~440MB), claude (~370MB).
- Action taken: ran inference inside `systemd-run --scope -p MemoryMax=1300M -p MemorySwapMax=0`
  bounded to 5 frames, so only the python process could ever be OOM-killed. Cap never tripped.
```

The earlier ONNX symlink has been removed. Use the real file `model/yolo11n.onnx`.

## TASK-007 YOLO RKNN/NPU detector — done (code 2026-06-11; NPU run 2026-06-13)

Deliverable code is complete and the NPU run is now validated. The original 2026-06-11 blocker is kept below as historical context.

Done:

- New `inference/yolo_rknn_detector.py` — same `detect() -> Target` interface, `last_timing`, and stale-hold behaviour as the ONNX detector; reuses `postprocess.letterbox` + `decode_yolo_output` + `select_target`. `_import_rknnlite()` tries `rknnlite.api` then `rknn_toolkit_lite2` and raises a clear `RuntimeError` — **no silent CPU fallback**. NPU input is letterboxed **NHWC uint8 RGB** (matches the `mean=0 / std=255` conversion default in `scripts/convert_yolo_rknn.sh`).
- `app.py` `finally` now calls a guarded `detector.close()` so the NPU runtime is released on exit.
- Confirmed OpenCV + ONNX detectors are unaffected by the new dispatch.

Historical blocker (resolved 2026-06-13):

```text
- rknn_toolkit_lite2 (import names: rknnlite / rknn_toolkit_lite2) NOT installed in env b
  -> both imports raise ModuleNotFoundError.
- No .rknn model present. Conversion (ONNX -> RKNN) is an x86-host step.
- Board has librknnrt.so + rknn_server + NPU driver v0.9.6 (TASK-006), so the
  hardware is fine; only the Python lite runtime + a converted model are missing.
```

Validation of the blocked path:

```text
_import_rknnlite()                      -> clean RuntimeError (lists both tried modules + install hint)
--detector yolo_rknn --model *.rknn     -> error: RKNN Lite runtime not available ... ; exit 1; no traceback; no p7 CSV
--detector face (regression)            -> still runs (detector="face")
```

Original next steps, now completed:

```text
1. x86 host: pip install rknn-toolkit2 (match the board runtime version), then
   bash scripts/convert_yolo_rknn.sh --onnx model/yolo11n.onnx \
       --output model/yolo11n.rknn --target rk3588
2. Board: install rknn_toolkit_lite2 matching NPU driver v0.9.6; copy the .rknn over.
3. Re-run the TASK-007 validation command; expect NPU inference + reports/logs/p7_yolo_rknn.csv.
```

Memory check (TASK-007): light path (import fails fast, no model load, no NPU). Before/after available ~1.7Gi, swap 0B.

Handoff (2026-06-11): board runtime is **librknnrt 1.4.0** (driver v0.9.6, py3.10/aarch64); the on-board `yolo11n.onnx` is opset 20 (too new), so re-exported a fixed-640 **opset-12** ONNX (cgroup-capped ultralytics export). Prepared `rknn_bundle/` (opset-12 `yolo11n.onnx` + `convert_yolo_rknn.py` + guide) and root `RKNN_NPU_GUIDE.md` for the x86 conversion. User converts on x86 → copies `yolo11n.rknn` back → installs `rknn_toolkit_lite2` → `bash scripts/run_rknn_demo.sh`.

### TASK-007 RKNN/NPU run — done (2026-06-13)

The remaining NPU blocker is resolved.

```text
Runtime: rknn-toolkit-lite2 / librknnrt 2.3.2, RKNPU driver 0.9.6
Model: model/yolo11n.rknn
20-frame run: reports/logs/p7_yolo_rknn.csv
  detector=yolo_rknn, target=person, found_frames=20/20
  avg_inference_ms=71.36, avg_fps=7.05
5-frame recheck: reports/logs/p7_yolo_rknn_recheck.csv
  found_frames=5/5, avg_inference_ms=114.45 (short warm-up run)
```

Validation command:

```bash
MODEL=model/yolo11n.rknn LOG=reports/logs/p7_yolo_rknn_recheck.csv MAX_FRAMES=5 \
  bash scripts/run_rknn_demo.sh --mock
```

Acceptance met: RKNN runtime initializes on the board, NPU inference completes, bbox/center/label/confidence are logged, and the detector does not fall back to CPU.

## TASK-010 demo orchestration scripts — done (2026-06-11)

Created 5 fixed-command demo scripts under `scripts/` (each: `set -euo pipefail`, cd to repo root, `/root/miniconda3/envs/b/bin/python3`, logs under `reports/logs/`, honor env overrides `CAMERA`/`DETECTOR`/`MODEL`/`TARGET_CLASS`/`WEB_HOST`/`WEB_PORT`/`PYTHON`/`MOCK`/`MAX_FRAMES`, print the final command, pass extra args through):

```text
run_opencv_demo.sh   run_web_demo.sh   run_onnx_demo.sh   run_rknn_demo.sh   benchmark_all.sh
```

Validation:

```text
bash -n x5                         -> all OK
run_opencv_demo.sh --mock          -> 保底 demo runs, detector=face
run_rknn_demo.sh --mock            -> now runs with model/yolo11n.rknn when runtime/model exist
benchmark_all.sh                   -> Python 740.6 ms/frame, pymp 273.2 ms/frame (2.71x),
                                      C ext 1.796 ms/frame (412.45x), max_diff <=4.6e-5;
                                      wrote reports/logs/cpp_conv_benchmark.txt
```

`benchmark_all.sh` has an opt-in `WITH_YOLO=1` branch that runs a short ONNX benchmark only if a model exists AND >=1.0GB is free (4GB memory guard). Memory during this task: light; available ~1.7Gi, swap 0B.

## TASK-011 README + final report — done (2026-06-11)

- `README.md`: intro now mentions the ONNX/RKNN detectors + built-in web monitor; layout adds `scripts/` + `model/`; new sections "One-line demo scripts", "YOLO detector modes", "Integrated web monitor", "Troubleshooting"; detector list + report pointer updated. Commands verified against current `app.py` flags.
- `reports/final_report.md`: created and later updated after RKNN/pymp completion. 10 sections (background → improvements) + requirement/bonus mapping + reproduce commands. Performance table cites real logs and now marks RKNN/NPU as completed.
- Real measured numbers used: OpenCV face ≈9.3 fps / ≈63 ms (p0); YOLO ONNX yolo11n ≈137 ms inference / ≈2.0 fps (p5); YOLO RKNN ≈71.4 ms inference / ≈7.0 fps (p7); pymp 5x5 conv 273.2 ms vs 740.6 ms → 2.71x; C 5x5 conv 1.796 ms vs 740.6 ms → 412.45x. TASK-008 real tuning is also completed in the later on-site section below.
- Memory: docs only, no heavy commands.

## TASK-008 real gimbal tuning — blocked (2026-06-11; mock prep done)

Real servo motion needs the board with PCA9685 + gimbal attached AND on-site human confirmation (proposal safety order: mock → calibrate → small max-step). The safe, no-hardware-risk parts are done; the real run is blocked pending a human at the board.

Done (safe, no I2C risk):

```text
mock-before-real smoke (30 frames): detector=face, average_fps 10.95, lost_frames 30 (headless),
  -> reports/logs/p8_mock_before_real.csv (31 rows)
mock calibrate (--mock-servo --calibrate): pan 95->85->90, tilt 95->85->90 (±5° sweep), exit 0
  -> validates the calibration code path without touching the bus
```

Blocked: `reports/logs/p8_real_tracking.csv` (needs real servos + human).

On-site procedure (do NOT change the order):

```text
1) python -m app --mock-servo --no-display --max-frames 30 --log reports/logs/p8_mock_before_real.csv
2) python -m app --calibrate                 # real ±5°, watch the limits
3) python -m app --detector face --max-step 1.0 --log reports/logs/p8_real_tracking.csv
4) axis moves AWAY from target -> flip --pan-sign / --tilt-sign
5) jitter -> raise --dead-zone; overshoot -> lower gain / max-step; STOP immediately on a limit hit
```

Recommended starting params (conservative config defaults; confirm direction on-site):

```text
pan-sign=+1 tilt-sign=-1 pan-gain=0.015 tilt-gain=0.015 dead-zone=25 max-step=1.0(first)/2.0
pan-range=45,135 tilt-range=60,120 center=90   (pan=ch0, tilt=ch1, bus=1, addr=0x40)
```

Memory check (TASK-008): light (mock only); available ~1.6Gi, swap 0B.

### TASK-008 real run — done (2026-06-11)

On the board with PCA9685 + gimbal attached, human present and confirming each step:

```text
I2C bus1: 0x40 (PCA9685) + 0x70 (all-call) detected; camera /dev/video0 free.
Real calibration (--calibrate): pan/tilt 90->95->85->90, smooth, no limit hit (user-confirmed).
Real tracking (face, --max-step 1.0, 150 frames -> reports/logs/p8_real_tracking.csv):
  16 fps; pan stayed 88-94, tilt 85-99; NO axis rode a limit (max consecutive at-limit = 0).
User confirmed: pan=horizontal, tilt=vertical, gimbal moves TOWARD the target.
=> signs correct (pan-sign=+1, tilt-sign=-1). Acceptance met (axes, direction, no limit hit).
```

CPU inference speed finding (motivates the NPU path):

```text
- Haar face found only 24/150 frames (16%) on a moving subject -> sluggish corrections (a
  detector-robustness issue, NOT a gimbal issue).
- yolo11n ONNX pure inference on CPU (cgroup-capped, 640 input):
    threads=4 -> 116.8 ms   threads=6 -> 132.4 ms   threads=8 -> 154.1 ms
  RK3588 big.LITTLE: extra A55 cores + sync overhead make MORE threads SLOWER. The detector's
  hardcoded 4 threads is already optimal; CPU floor ~117 ms/inference at 640.
- => CPU thread tuning is a dead end. Real speedups: NPU/RKNN (TASK-007, ~15-40 ms expected) or
  a smaller input (imgsz 320/416, ~1/4 FLOPs at 320). Concrete motivation for the NPU bonus.
```

## root layout finalization — done (2026-06-14 18:09:16 CST)

- Updated the package layout again so the real package now lives directly at the repository root: [app.py](/root/ws/app.py), [camera.py](/root/ws/camera.py), [core](/root/ws/core), [inference](/root/ws/inference), [motion](/root/ws/motion), [web](/root/ws/web), and [voice](/root/ws/voice). Imports resolve either by editable install or by repo-local scripts exporting `PYTHONPATH=$ROOT`.
- Kept the root [pyproject.toml](/root/ws/pyproject.toml) added earlier so the tracker installs cleanly with `python -m pip install -e . --no-deps`.
- Moved tracker tests to top-level [tests/test_postprocess.py](/root/ws/tests/test_postprocess.py) and [tests/test_voice.py](/root/ws/tests/test_voice.py), and updated pytest discovery to `testpaths = ["tests"]`.
- Updated repo-local entry points that used to rely on the deleted compatibility package. The old `servo/servo_test.py` compatibility wrapper has since been removed; use [scripts/servo_axis_sweep.sh](/root/ws/scripts/servo_axis_sweep.sh) for direct servo channel tests. Shell wrappers under `scripts/` add the repo root explicitly when they run from the repo checkout. The camera-only web stream is available through `gimbal-tracker-stream-webcam` or `python -m web.stream_webcam`.
- Updated repo docs/instructions ([README.md](/root/ws/README.md), [AGENTS.md](/root/ws/AGENTS.md)) so they describe the current root-level package layout and no longer point to a non-existent main `src/` directory.

Validation:

```text
python -m pip install -e . --no-deps                    -> OK
cd /tmp && python -c "import core.config"               -> imports from core/config.py after editable install
cd /tmp && gimbal-tracker --help                        -> OK
cd /tmp && gimbal-tracker-stream-webcam --help          -> OK
cd /tmp && gimbal-tracker-tune-detector --help          -> OK
python tests/test_postprocess.py                        -> 8/8 passed
python tests/test_voice.py                              -> 4/4 passed
find . -maxdepth 2 ...                                  -> package lives directly at repo root plus tests
```

Note: `python -m pytest tests -q` is still unavailable in env `b` because `pytest` is not installed there; this is unchanged from earlier tasks, and the direct-run test harness remains the reliable validation path.

## legacy bonus archive — done (2026-06-14 18:09:16 CST)

- Moved the C acceleration module to [acceleration](/root/ws/acceleration), including `benchmark.py`, `py_conv.py`, `setup.py`, `src/cconv.c`, the built `_cconv` extension, and wheel artifacts under `dist/`.
- Updated [scripts/benchmark_all.sh](/root/ws/scripts/benchmark_all.sh) and [scripts/benchmark_nms.sh](/root/ws/scripts/benchmark_nms.sh) to use the `acceleration/` location.
- Updated [inference/postprocess.py](/root/ws/inference/postprocess.py) so optional C NMS acceleration loads the installed `cpp_conv` package first, then falls back to the local extension under `acceleration/`.

## data symlink cleanup — done (2026-06-14 18:17:00 CST)

- Removed the `.data -> data` compatibility symlink. Runtime defaults and scripts already use `data/` directly.
- Updated README/agent docs to remove `.data/` as a documented compatibility path.

## post-cleanup dry-run — partial (2026-06-14 18:37:37 CST)

- `python tests/test_postprocess.py` -> 8/8 passed.
- `python tests/test_voice.py` -> 4/4 passed.
- `python -m app --detector yolo_onnx --model model/yolo11n.onnx --mock-servo --no-display --max-frames 1` -> passed, printed a JSON summary.
- `python -m app --mock-servo --no-display --max-frames 1` -> still segfaults in OpenCV Haar detection (`inference/detector.py::_detect_boxes`).
- `python -m app --detector yolo_rknn --model model/yolo11n.rknn --mock-servo --no-display --max-frames 1` -> model/runtime loaded, then failed only because `/dev/video0` was not available in this environment.

Known remaining issue outside the migration work:

```text
python -m app --mock-servo --no-display --max-frames 1
```

still segfaults inside OpenCV Haar detection on this board (trace lands in
`inference/detector.py::_detect_boxes`). This is a runtime /
OpenCV stability problem, not a `src` packaging/import problem.
