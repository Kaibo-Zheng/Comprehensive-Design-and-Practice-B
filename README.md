# Two-Axis Gimbal Person Tracker

This is the course project directory for a two-axis camera gimbal person tracker.
The main demo uses OpenCV face/person detection, a proportional servo controller,
CSV metrics logging, and a PCA9685-driven two-axis gimbal. Detection is
pluggable behind one `Target` interface: lightweight OpenCV
(`face`/`upperbody`/`hog`), a YOLO **ONNX** detector on the CPU, and a YOLO
**RKNN** detector for the RK3588 NPU. An optional built-in web monitor streams
the annotated video and live status. C/C++ acceleration code lives under
`acceleration/`.

## Directory Layout

```text
.
├── app.py                 # Main application entry point
├── camera.py              # Camera access
├── core/                  # Config and metrics
├── inference/             # OpenCV / YOLO detectors and postprocess
├── motion/                # Controller and gimbal actuation
├── web/                   # Web monitor and MJPEG streamer
├── voice/                 # Voice notification
├── tests/                 # Tracker unit tests
├── acceleration/          # C/C++ acceleration code and benchmark package
│   └── src/cconv.c        # C extension source
├── archive/
│   ├── legacy_experiments/
│   ├── legacy_weights/
│   └── legacy_bonus/
│       └── rknn_bundle/   # archived RKNN/NPU conversion toolchain and notes
├── scripts/               # Fixed one-line demo / convert / benchmark commands
│   └── run.sh             # Final integrated hardware demo
├── model/                 # Model checkpoints and conversion notes
├── audio/                 # Playback-mode voice alert WAV assets
├── illustration/          # Generated figures and screenshots
├── result/                # Final report, task summary, runtime logs
├── reports/               # Compatibility links to result/audio/illustration
├── doc/                   # Local notes and supporting documents
├── AGENTS.md              # Repository instructions
├── pyproject.toml         # Editable install / console entry points
├── 大作业要求.pdf          # Assignment requirements
└── environment.yml        # Conda environment recipe
```

## Environment

The active project environment is the conda environment `b`.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate b
```

To recreate or update it on this board from the checked-in environment file:

```bash
conda env update -n b -f environment.yml
conda activate b
python -m pip install -e . --no-deps
python -m pip check
```

`environment.yml` records the package set used on this board. The editable
install provides the `gimbal-tracker*` console entry points while leaving the
repo-local `python -m ...` commands unchanged.

The Python modules and packages live directly at the repo root. Repo-local
scripts add the repo root to `PYTHONPATH`, and external use should go through
an editable install.

For just refreshing the editable install:

```bash
python -m pip install -e . --no-deps
```

## Main Commands

Mock-servo smoke test:

```bash
python -m app --mock-servo --no-display --max-frames 5 --log result/logs/tracking_smoke.csv
```

Detector preview:

```bash
python -m inference.tune_detector --detector face
```

Webcam MJPEG stream for SSH viewing:

```bash
/root/miniconda3/envs/b/bin/python3 -m web.stream_webcam --camera 0 --host 127.0.0.1 --port 8080 --width 640 --height 480 --fps 30
```

From your local machine, forward the port over SSH:

```bash
ssh -L 8080:127.0.0.1:8080 user@board-ip
```

Then open `http://127.0.0.1:8080/`. The stream endpoint is `/stream.mjpg`,
single-frame snapshot is `/snapshot.jpg`, and runtime status is `/status.json`.
If testing with `curl` on this board, use `curl --noproxy 127.0.0.1 ...` so the
local proxy does not intercept the loopback request.

Safe calibration:

```bash
python -m app --mock-servo --calibrate
python -m app --calibrate
```

Real gimbal run:

```bash
python -m app --detector face --log result/logs/tracking_log.csv
```

Acceleration benchmarks (pure Python, `pymp`, and C extension):

```bash
python acceleration/benchmark.py --height 160 --width 240 --iterations 3
bash scripts/benchmark_nms.sh --boxes 1200 --iterations 50
```

Build the C extension wheel:

```bash
python -m build acceleration
```

## One-line demo scripts

Fixed entry points under `scripts/` so live demos don't depend on long command
lines. Each prints the exact command it runs and writes a CSV under
`result/logs/`. Override defaults with env vars (`CAMERA`, `MODEL`, `WEB_PORT`,
`MOCK=1`, `MAX_FRAMES`, ...).

```bash
bash scripts/run_opencv_demo.sh --mock   # fallback OpenCV demo, headless mock servo
bash scripts/run_opencv_demo.sh          # OpenCV demo, real servo (on the board)
bash scripts/run_web_demo.sh             # tracking + web monitor (browse the board)
bash scripts/run_voice_demo.sh --mock    # target acquisition voice notification
bash scripts/run_onnx_demo.sh --mock     # YOLO ONNX CPU demo
bash scripts/run_rknn_demo.sh --mock     # YOLO RKNN NPU demo
bash scripts/benchmark_all.sh            # Python/pymp/C 5x5 conv benchmark (WITH_YOLO=1 adds ONNX)
bash scripts/benchmark_nms.sh            # Python vs C YOLO NMS benchmark
```

## YOLO detector modes

Detection is chosen with `--detector`. YOLO needs a model via `--model`; the COCO
`person` class is tracked by default. A missing model or missing NPU runtime exits
with a clear error — never a silent fallback to another device.

```bash
# YOLO ONNX on the CPU (onnxruntime)
python -m app --detector yolo_onnx \
  --model model/yolo11n.onnx --target-class person \
  --mock-servo --no-display --max-frames 30 --log result/logs/p5_yolo_onnx.csv

# YOLO RKNN on the RK3588 NPU (rknnlite + converted .rknn)
python -m app --detector yolo_rknn \
  --model model/yolo11n.rknn --target-class person \
  --mock-servo --no-display --max-frames 30 --log result/logs/p7_yolo_rknn.csv
```

Convert ONNX → RKNN on an x86 host (the board only runs the result):

```bash
bash scripts/convert_yolo_rknn.sh --onnx model/yolo11n.onnx \
  --output model/yolo11n.rknn --target rk3588
```

Models are **not** committed; the current board keeps the RKNN demo model under
`model/yolo11n.rknn`. See `model/README.md` for sourcing/conversion.

## Final integrated demo

This is the full live command for the final hardware demo: RKNN/NPU person
detection, closed-loop gimbal tracking, integrated Web monitor, and Chinese
voice notification on target acquisition. The wrapper script also pins
CPU/NPU/DDR governors to `performance` and limits the MJPEG stream to reduce
Web encoding overhead.

```bash
cd /root/ws && bash scripts/run.sh
```

For the physical gimbal, `scripts/run.sh` uses conservative tilt defaults to avoid
snapping the second servo into a mechanical corner: `TILT_CENTER=45`,
`TILT_RANGE=0,90`, `PAN_SIGN=-1`, `TILT_SIGN=1`, `MAX_STEP=0.8`, and
`STARTUP_CENTER=pan`.
That startup mode centers only the pan axis; the tilt axis is written only when
vertical target error requires it. Tune them without editing the script:

```bash
TILT_CENTER=45 TILT_RANGE=0,90 MAX_STEP=0.8 bash scripts/run.sh
```

To avoid any startup servo movement, use `STARTUP_CENTER=none bash scripts/run.sh`.

Open `http://<board-ip>:8080/` from a browser. The default voice playback
script sends audio to USB Audio device `plughw:3,0`; set
`VOICE_DEVICE=plughw:2,0` (or another ALSA device) before the command if the
speaker is connected to a different sound card.

## Integrated web monitor

`app.py` can serve the annotated video and live status itself (no extra process) —
the recommended way to watch a headless run:

```bash
python -m app --detector face --web-host 0.0.0.0 --web-port 8080 \
  --no-display --log result/logs/web_demo.csv
```

Endpoints: `/` (page), `/stream.mjpg`, `/snapshot.jpg`, `/status.json`. The
standalone `web.stream_webcam` module above is a camera-only
alternative.

## Voice notification

The tracker can optionally play a short voice notification when a target is
first acquired. It is off by default, runs in a background thread, and has a
cooldown so one continuous detection does not speak every frame. The default
announcement is Chinese (`检测到人员。`, from `--voice-text "检测到{label_zh}。"`);
pass `--voice-text "Detected {label}."` and point `--voice-audio-path` at a
matching WAV if you want English instead.

On this 4 GB board, the repo is configured around **playback mode** for voice
alerts. Use one of the bundled WAV files or provide your own:

```bash
python -m app --detector face --mock-servo --no-display \
  --voice-enabled --voice-mode play \
  --voice-audio-path audio/found_person_zh_beep_loud_44k.wav \
  --voice-cooldown 5
```

Useful options:

- `--voice-mode play`: play a prepared WAV without loading MOSS TTS during tracking.
- `--voice-audio-path audio/found_person_zh_beep_loud_44k.wav`: current demo WAV (`发现人员。`) with a short wake-up tone before the Chinese announcement.
- `--voice-player "scripts/play_voice_usb.sh"`: play via USB Audio (`VOICE_DEVICE` overrides the ALSA device).
- `--voice-no-playback`: exercise event logic without using the audio device.

## Troubleshooting

- **Camera busy / won't open**: `ls -l /dev/video*`, `fuser -v /dev/video0`; stop
  the old process rather than blindly changing the camera index.
- **RKNN runtime missing**: install `rknn_toolkit_lite2` on the board and provide a
  `.rknn` model, or use `--detector yolo_onnx`. Collect board status with
  `bash scripts/check_rknn_env.sh`.
- **Port 8080 busy**: use a different `--web-port` (or `WEB_PORT=...` for the script).
- **Servo moves the wrong way**: flip `--pan-sign` / `--tilt-sign`; reduce
  `--max-step` and widen `--dead-zone` if it jitters.
- **One servo does not physically move**: stop the tracker first, then test the
  PCA9685 channel directly, for example
  `bash scripts/servo_axis_sweep.sh --axis tilt --channel 1 --center 90 --delta 5`.
- **Tilt servo jumps to a bad angle on startup**: lower `TILT_CENTER` and keep
  `TILT_RANGE` narrow for the current mechanical mount, for example
  `TILT_CENTER=45 TILT_RANGE=0,90 bash scripts/run.sh`.
- **Tilt follows away from the target**: flip the final-demo sign with
  `TILT_SIGN=-1 bash scripts/run.sh` or `TILT_SIGN=1 bash scripts/run.sh`; the right value
  depends on the current servo horn orientation.
- **Low memory (~4GB board)**: don't run YOLO inference, the web service, and a
  benchmark at the same time; prefer `--mock-servo --no-display --max-frames 30`
  for tests and check `free -h` before/after heavy commands.
- **Voice is silent**: confirm `aplay` can use the audio device. For YOLO runs,
  use `--voice-mode play` with a prepared WAV; runtime MOSS synthesis can exceed
  memory headroom on this board.

## Notes

- Run real servo commands only on the board with the PCA9685 and gimbal connected.
- The default servo channels are pan `0` and tilt `1`.
- Detectors: `face` (default, OpenCV Haar), `upperbody`, `hog`, `yolo_onnx` (CPU), `yolo_rknn` (RK3588 NPU).
- Logs used in the report are under `result/logs/`; the per-task summary is in `result/result.md`, and the final report is `result/final_report.md`.
- `archive/` is local backup material and is not required for submission.
