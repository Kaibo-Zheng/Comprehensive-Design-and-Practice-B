"""实时云台人体跟踪主程序。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Tuple

import cv2

from tracker.camera import open_camera
from common.config import (
    CameraConfig,
    ControllerConfig,
    DetectorConfig,
    ServoAxisConfig,
    ServoConfig,
    VoiceConfig,
    parse_camera_source,
)
from common.metrics import FrameTimings, MetricsSnapshot, TrackingMetrics
from inference.detector import PersonDetector, Target
from motion import Gimbal, TrackingController
from voice.notifier import VoiceNotifier
from web.monitor import WebMonitor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-axis gimbal person tracker")
    parser.add_argument("--camera", type=parse_camera_source, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--detector", choices=["face", "upperbody", "hog", "yolo_onnx", "yolo_rknn"], default="face")
    parser.add_argument("--cascade-path", default=None)
    parser.add_argument("--hold-frames", type=int, default=5)
    parser.add_argument("--model", default=None, help="Path to YOLO .onnx / .rknn model (yolo_* detectors)")
    parser.add_argument("--target-class", default="person", help="COCO class to track for YOLO detectors")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--input-size", type=int, default=640)

    parser.add_argument("--mock-servo", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument(
        "--startup-center",
        choices=["both", "pan", "tilt", "none"],
        default="both",
        help="Which axis to move to center at startup before tracking begins",
    )
    parser.add_argument("--bus", type=int, default=1)
    parser.add_argument("--address", type=lambda x: int(x, 0), default=0x40)
    parser.add_argument("--pan-channel", type=int, default=0)
    parser.add_argument("--tilt-channel", type=int, default=1)
    parser.add_argument("--pan-range", default="45,135")
    parser.add_argument("--tilt-range", default="60,120")
    parser.add_argument(
        "--center",
        type=float,
        default=90.0,
        help="Shared default center angle for both axes unless per-axis centers are set",
    )
    parser.add_argument("--pan-center", type=float, default=None, help="Pan center angle; defaults to --center")
    parser.add_argument("--tilt-center", type=float, default=None, help="Tilt center angle; defaults to --center")
    parser.add_argument("--pulse-min", type=int, default=150)
    parser.add_argument("--pulse-max", type=int, default=600)

    parser.add_argument("--pan-gain", type=float, default=0.015)
    parser.add_argument("--tilt-gain", type=float, default=0.015)
    parser.add_argument("--pan-sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--tilt-sign", type=int, choices=[-1, 1], default=-1)
    parser.add_argument("--dead-zone", type=int, default=25)
    parser.add_argument("--max-step", type=float, default=2.0)

    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--web-host", default=None, help="Bind address for the web monitor, e.g. 0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--web-quality", type=int, default=80, help="JPEG quality (1-100) for the web stream")
    parser.add_argument(
        "--web-max-fps",
        type=float,
        default=0.0,
        help="Maximum MJPEG frame rate for the web stream; 0 means encode every tracked frame",
    )
    parser.add_argument("--log", default=None)
    parser.add_argument("--max-frames", type=int, default=0)

    parser.add_argument("--voice-enabled", action="store_true", help="Speak once when a target is acquired")
    parser.add_argument(
        "--voice-mode",
        choices=["play", "synthesize"],
        default="play",
        help="play uses a prepared WAV; synthesize loads MOSS TTS at runtime",
    )
    parser.add_argument("--voice-text", default="妫€娴嬪埌{label_zh}銆?")
    parser.add_argument("--voice-cooldown", type=float, default=5.0, help="Minimum seconds between voice events")
    parser.add_argument("--voice-audio-path", default=None, help="Prepared WAV to play for every voice event")
    parser.add_argument("--voice-model-dir", default=None, help="External MOSS TTS model directory for synthesize mode")
    parser.add_argument("--voice-runtime-dir", default=None, help="External MOSS TTS runtime directory for synthesize mode")
    parser.add_argument("--voice-output-dir", default="audio")
    parser.add_argument("--voice-preset", default="Junhao")
    parser.add_argument("--voice-sample-mode", choices=["greedy", "fixed", "full"], default="greedy")
    parser.add_argument("--voice-max-new-frames", type=int, default=24)
    parser.add_argument("--voice-threads", type=int, default=2)
    parser.add_argument("--voice-player", default="aplay")
    parser.add_argument("--voice-seed", type=int, default=1)
    parser.add_argument("--voice-enable-wetext", action="store_true")
    parser.add_argument("--voice-no-playback", action="store_true", help="Generate audio files but do not play them")
    return parser.parse_args(argv)


def parse_range(value: str, name: str) -> Tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"{name} must be formatted as min,max")
    low, high = float(parts[0]), float(parts[1])
    if low >= high:
        raise argparse.ArgumentTypeError(f"{name} minimum must be less than maximum")
    return low, high


def build_configs(
    args: argparse.Namespace,
) -> tuple[CameraConfig, DetectorConfig, ServoConfig, ControllerConfig, VoiceConfig]:
    pan_min, pan_max = parse_range(args.pan_range, "--pan-range")
    tilt_min, tilt_max = parse_range(args.tilt_range, "--tilt-range")
    pan_center = args.center if args.pan_center is None else args.pan_center
    tilt_center = args.center if args.tilt_center is None else args.tilt_center
    if not pan_min <= pan_center <= pan_max:
        raise argparse.ArgumentTypeError("--pan-center must be within --pan-range")
    if not tilt_min <= tilt_center <= tilt_max:
        raise argparse.ArgumentTypeError("--tilt-center must be within --tilt-range")

    camera_config = CameraConfig(source=args.camera, width=args.width, height=args.height, fps=args.fps)
    detector_config = DetectorConfig(
        detector=args.detector,
        hold_frames=args.hold_frames,
        cascade_path=args.cascade_path,
        model_path=args.model,
        target_class=args.target_class,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        input_size=args.input_size,
    )
    servo_config = ServoConfig(
        bus_number=args.bus,
        address=args.address,
        pan=ServoAxisConfig(args.pan_channel, pan_min, pan_max, pan_center),
        tilt=ServoAxisConfig(args.tilt_channel, tilt_min, tilt_max, tilt_center),
        pulse_min=args.pulse_min,
        pulse_max=args.pulse_max,
    )
    controller_config = ControllerConfig(
        pan_gain=args.pan_gain,
        tilt_gain=args.tilt_gain,
        pan_sign=args.pan_sign,
        tilt_sign=args.tilt_sign,
        dead_zone=args.dead_zone,
        max_step=args.max_step,
    )
    voice_config = VoiceConfig(
        enabled=args.voice_enabled,
        mode=args.voice_mode,
        text=args.voice_text,
        cooldown_s=args.voice_cooldown,
        audio_path=args.voice_audio_path,
        model_dir=args.voice_model_dir,
        runtime_dir=args.voice_runtime_dir,
        output_dir=args.voice_output_dir,
        voice_preset=args.voice_preset,
        sample_mode=args.voice_sample_mode,
        max_new_frames=args.voice_max_new_frames,
        threads=args.voice_threads,
        playback=not args.voice_no_playback,
        player=args.voice_player,
        seed=args.voice_seed,
        enable_wetext=args.voice_enable_wetext,
    )
    return camera_config, detector_config, servo_config, controller_config, voice_config


def draw_overlay(
    frame,
    target: Target | None,
    result,
    metrics: MetricsSnapshot,
) -> None:
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    cv2.drawMarker(frame, center, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1)

    if target is not None:
        x, y, w, h = target.bbox
        color = (0, 255, 255) if target.stale else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, target.center, 4, color, -1)
        cv2.line(frame, center, target.center, color, 1)
        label = "stale" if target.stale else target.label
        cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        cv2.putText(frame, "target lost", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    lines = [
        f"FPS: {metrics.fps:.1f}",
        f"err: ({result.x_error}, {result.y_error}) px",
        f"pan/tilt: {result.pan:.1f}/{result.tilt:.1f}",
        f"lost: {metrics.lost_frames}",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(frame, line, (20, height - 90 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def build_detector(config: DetectorConfig):
    """根据配置构造对应的检测器。

    YOLO 模块采用延迟导入，因此只运行 OpenCV 检测时不需要安装
    onnxruntime 或 rknnlite。
    """

    if config.detector in {"face", "upperbody", "hog"}:
        return PersonDetector(config)
    if config.detector == "yolo_onnx":
        from inference.yolo_onnx_detector import YoloOnnxDetector
        return YoloOnnxDetector(config)
    if config.detector == "yolo_rknn":
        from inference.yolo_rknn_detector import YoloRknnDetector
        return YoloRknnDetector(config)
    raise ValueError(f"Unsupported detector: {config.detector}")


def run(args: argparse.Namespace) -> int:
    camera_config, detector_config, servo_config, controller_config, voice_config = build_configs(args)
    if voice_config.enabled and voice_config.mode == "synthesize" and detector_config.detector.startswith("yolo_"):
        raise RuntimeError(
            "Runtime voice synthesis with YOLO is disabled on this low-memory board. "
            "Use playback mode with an existing WAV via "
            "--voice-mode play --voice-audio-path <wav>."
        )
    gimbal = Gimbal(servo_config, mock=args.mock_servo)

    if args.calibrate:
        gimbal.safe_calibration()
        gimbal.close()
        return 0

    detector = build_detector(detector_config)
    controller = TrackingController(controller_config, servo_config)
    metrics = TrackingMetrics(args.log, detector=args.detector)
    cap = open_camera(camera_config)
    voice: VoiceNotifier | None = VoiceNotifier(voice_config, args.detector) if voice_config.enabled else None

    monitor: WebMonitor | None = None
    if args.web_host:
        monitor = WebMonitor(args.web_host, args.web_port)
        monitor.start()
        print(f"web monitor at http://{args.web_host}:{args.web_port}/", flush=True)
    web_encode_params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, args.web_quality))]
    web_min_interval = 0.0 if args.web_max_fps <= 0 else 1.0 / args.web_max_fps
    last_web_frame_at = 0.0

    try:
        if args.startup_center == "both":
            gimbal.center()
        elif args.startup_center in {"pan", "tilt"}:
            gimbal.center_axis(args.startup_center)
        while True:
            frame_start = time.perf_counter()

            t0 = time.perf_counter()
            ok, frame = cap.read()
            capture_ms = (time.perf_counter() - t0) * 1000.0
            if not ok:
                raise RuntimeError("Failed to read frame from camera")

            t0 = time.perf_counter()
            target = detector.detect(frame)
            detect_ms = (time.perf_counter() - t0) * 1000.0
            if voice is not None:
                voice.update(target)

            control_center = target.center if target is not None and not target.stale else None
            result = controller.update(
                control_center,
                frame_size=(frame.shape[1], frame.shape[0]),
            )
            if result.pan_moved:
                gimbal.set_angle("pan", result.pan)
            if result.tilt_moved:
                gimbal.set_angle("tilt", result.tilt)

            # 检测器可以通过 last_timing 提供预处理、推理和后处理耗时；
            # 若没有提供拆分结果，则把整个 detect() 调用耗时记为推理耗时。
            stage = getattr(detector, "last_timing", None)
            if stage:
                timings = FrameTimings(
                    capture_ms=capture_ms,
                    preprocess_ms=float(stage.get("preprocess_ms", 0.0)),
                    inference_ms=float(stage.get("inference_ms", 0.0)),
                    postprocess_ms=float(stage.get("postprocess_ms", 0.0)),
                    total_ms=(time.perf_counter() - frame_start) * 1000.0,
                )
            else:
                timings = FrameTimings(
                    capture_ms=capture_ms,
                    inference_ms=detect_ms,
                    total_ms=(time.perf_counter() - frame_start) * 1000.0,
                )

            snapshot = metrics.update(
                found=target is not None,
                stale=bool(target and target.stale),
                x_error=result.x_error,
                y_error=result.y_error,
                pan=result.pan,
                tilt=result.tilt,
                label=(target.label if target is not None else ""),
                confidence=(target.confidence if target is not None else None),
                timings=timings,
            )

            if not args.no_display:
                draw_overlay(frame, target, result, snapshot)
                cv2.imshow("gimbal person tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if monitor is not None:
                now = time.perf_counter()
                jpeg = None
                if web_min_interval == 0.0 or now - last_web_frame_at >= web_min_interval:
                    overlay = frame.copy()
                    draw_overlay(overlay, target, result, snapshot)
                    ok_jpeg, encoded = cv2.imencode(".jpg", overlay, web_encode_params)
                    if ok_jpeg:
                        jpeg = encoded.tobytes()
                        last_web_frame_at = now
                monitor.update(
                    jpeg,
                    {
                        "detector": args.detector,
                        "fps": snapshot.fps,
                        "frame_index": snapshot.frame,
                        "target_found": target is not None,
                        "target_stale": bool(target and target.stale),
                        "target_label": target.label if target is not None else "",
                        "target_confidence": (
                            round(target.confidence, 4)
                            if target is not None and target.confidence is not None
                            else None
                        ),
                        "x_error": result.x_error,
                        "y_error": result.y_error,
                        "error_norm": round(snapshot.error_px, 2),
                        "pan": result.pan,
                        "tilt": result.tilt,
                        "lost_frames": snapshot.lost_frames,
                    },
                )

            if args.max_frames and metrics.frame_count >= args.max_frames:
                break

    finally:
        cap.release()
        detector_close = getattr(detector, "close", None)
        if callable(detector_close):
            detector_close()
        gimbal.close()
        metrics.close()
        if voice is not None:
            voice.close()
        if monitor is not None:
            monitor.stop()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(json.dumps(metrics.summary(), ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
