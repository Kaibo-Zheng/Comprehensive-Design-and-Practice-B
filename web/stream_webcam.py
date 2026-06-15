#!/usr/bin/env python3
"""通过 HTTP 将 OpenCV 摄像头画面输出为 MJPEG 视频流。

主要用于配合 SSH 端口转发查看开发板摄像头画面：

    python -m web.stream_webcam --camera 0 --host 127.0.0.1 --port 8080
    ssh -L 8080:127.0.0.1:8080 user@board

随后在本机浏览器打开对应地址即可。
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def parse_camera_source(value: str) -> int | str:
    if value.isdigit():
        return int(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream a local OpenCV webcam as MJPEG over HTTP.")
    parser.add_argument("--camera", type=parse_camera_source, default=0, help="Camera index or device path.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address. Use 127.0.0.1 with SSH tunneling.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="MJPG", help="FourCC requested from the camera, or empty to leave unset.")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality from 1 to 100.")
    parser.add_argument(
        "--backend",
        choices=("auto", "v4l2", "default"),
        default="v4l2",
        help="OpenCV capture backend. v4l2 is best for Linux USB webcams.",
    )
    return parser.parse_args()


class FrameStore:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.frame_id = 0
        self.frames_read = 0
        self.encode_errors = 0
        self.read_errors = 0
        self.started_at = time.time()
        self.last_frame_at: float | None = None
        self.last_error: str | None = None
        self.camera_opened = False
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0

    def mark_opened(self, width: int, height: int, fps: float) -> None:
        with self.condition:
            self.camera_opened = True
            self.actual_width = width
            self.actual_height = height
            self.actual_fps = fps
            self.last_error = None
            self.condition.notify_all()

    def set_error(self, message: str) -> None:
        with self.condition:
            self.last_error = message
            self.condition.notify_all()

    def update_frame(self, jpeg: bytes) -> None:
        with self.condition:
            self.jpeg = jpeg
            self.frame_id += 1
            self.frames_read += 1
            self.last_frame_at = time.time()
            self.last_error = None
            self.condition.notify_all()

    def note_read_error(self) -> None:
        with self.condition:
            self.read_errors += 1
            self.last_error = "failed to read frame from camera"
            self.condition.notify_all()

    def note_encode_error(self) -> None:
        with self.condition:
            self.encode_errors += 1
            self.last_error = "failed to encode frame as JPEG"
            self.condition.notify_all()

    def wait_for_frame(self, last_seen: int, timeout: float = 2.0) -> tuple[int, bytes | None]:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.frame_id == last_seen and not self.jpeg:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self.frame_id, self.jpeg
                self.condition.wait(remaining)
            while self.frame_id == last_seen:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self.frame_id, self.jpeg
                self.condition.wait(remaining)
            return self.frame_id, self.jpeg

    def snapshot(self, timeout: float = 5.0) -> bytes | None:
        with self.condition:
            if self.jpeg is None:
                self.condition.wait(timeout)
            return self.jpeg

    def wait_for_camera(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.camera_opened and self.last_error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(remaining)
            return self.camera_opened

    def status(self, clients: int, camera: int | str, requested: dict[str, Any]) -> dict[str, Any]:
        with self.condition:
            now = time.time()
            age = None if self.last_frame_at is None else max(0.0, now - self.last_frame_at)
            elapsed = max(0.001, now - self.started_at)
            if not self.camera_opened and self.last_error:
                state = "error"
            elif self.last_frame_at is None:
                state = "waiting_for_frames"
            else:
                state = "running"
            return {
                "state": state,
                "camera": camera,
                "requested": requested,
                "actual": {
                    "width": self.actual_width,
                    "height": self.actual_height,
                    "fps": self.actual_fps,
                },
                "frames": self.frames_read,
                "stream_frames": self.frame_id,
                "capture_fps_observed": self.frames_read / elapsed,
                "clients": clients,
                "last_frame_age_s": age,
                "read_errors": self.read_errors,
                "encode_errors": self.encode_errors,
                "last_error": self.last_error,
            }


class WebcamCapture(threading.Thread):
    def __init__(self, args: argparse.Namespace, store: FrameStore, stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self.args = args
        self.store = store
        self.stop_event = stop_event
        self.capture = None

    def run(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            self.store.set_error(f"OpenCV import failed: {exc}")
            self.stop_event.set()
            return

        backend = self._backend_value(cv2)
        if backend is None:
            capture = cv2.VideoCapture(self.args.camera)
        else:
            capture = cv2.VideoCapture(self.args.camera, backend)
        self.capture = capture

        if self.args.fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.args.fourcc[:4]))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        capture.set(cv2.CAP_PROP_FPS, self.args.fps)

        if not capture.isOpened():
            self.store.set_error(f"failed to open camera source {self.args.camera!r}")
            self.stop_event.set()
            return

        self.store.mark_opened(
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(capture.get(cv2.CAP_PROP_FPS)),
        )
        quality = max(1, min(100, self.args.quality))
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

        try:
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    self.store.note_read_error()
                    time.sleep(0.02)
                    continue

                ok, encoded = cv2.imencode(".jpg", frame, encode_params)
                if not ok:
                    self.store.note_encode_error()
                    continue
                self.store.update_frame(encoded.tobytes())
        finally:
            capture.release()

    def _backend_value(self, cv2: Any) -> int | None:
        if self.args.backend == "default":
            return None
        if self.args.backend == "v4l2":
            return cv2.CAP_V4L2
        return None


class StreamingServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        store: FrameStore,
        stop_event: threading.Event,
        args: argparse.Namespace,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.store = store
        self.stop_event = stop_event
        self.args = args
        self.client_count = 0
        self.client_lock = threading.Lock()

    def add_client(self) -> None:
        with self.client_lock:
            self.client_count += 1

    def remove_client(self) -> None:
        with self.client_lock:
            self.client_count = max(0, self.client_count - 1)

    def clients(self) -> int:
        with self.client_lock:
            return self.client_count


class StreamHandler(BaseHTTPRequestHandler):
    server: StreamingServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send_index()
        elif path == "/stream.mjpg":
            self._send_stream()
        elif path == "/snapshot.jpg":
            self._send_snapshot()
        elif path == "/status.json":
            self._send_status()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def _send_index(self) -> None:
        body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Webcam Stream</title>
  <style>
    body {{ margin: 0; background: #151515; color: #f2f2f2; font-family: Arial, sans-serif; }}
    header {{ padding: 12px 16px; background: #222; display: flex; justify-content: space-between; gap: 16px; }}
    main {{ display: grid; place-items: center; min-height: calc(100vh - 48px); }}
    img {{ width: min(100vw, 960px); height: auto; display: block; }}
    a {{ color: #8fd3ff; }}
  </style>
</head>
<body>
  <header>
    <strong>Webcam Stream</strong>
    <span><a href="/snapshot.jpg">snapshot</a> | <a href="/status.json">status</a></span>
  </header>
  <main><img src="/stream.mjpg" alt="live webcam stream"></main>
</body>
</html>
"""
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_status(self) -> None:
        requested = {
            "width": self.server.args.width,
            "height": self.server.args.height,
            "fps": self.server.args.fps,
            "fourcc": self.server.args.fourcc,
            "quality": self.server.args.quality,
            "backend": self.server.args.backend,
        }
        status = self.server.store.status(self.server.clients(), self.server.args.camera, requested)
        data = json.dumps(status, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_snapshot(self) -> None:
        jpeg = self.server.store.snapshot()
        if jpeg is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "no frame available yet")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(jpeg)

    def _send_stream(self) -> None:
        self.server.add_client()
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()

        last_seen = 0
        try:
            while not self.server.stop_event.is_set():
                frame_id, jpeg = self.server.store.wait_for_frame(last_seen)
                if jpeg is None or frame_id == last_seen:
                    continue
                last_seen = frame_id
                self.wfile.write(b"--FRAME\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass
        finally:
            self.server.remove_client()


def run(args: argparse.Namespace) -> int:
    store = FrameStore()
    stop_event = threading.Event()
    capture = WebcamCapture(args, store, stop_event)
    capture.start()

    if not store.wait_for_camera():
        stop_event.set()
        capture.join(timeout=2.0)
        error = store.last_error or f"timed out opening camera source {args.camera!r}"
        print(f"error: {error}", file=sys.stderr)
        return 1

    server = StreamingServer((args.host, args.port), StreamHandler, store, stop_event, args)

    def handle_signal(_signum: int, _frame: Any) -> None:
        stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(
        f"Serving camera {args.camera!r} at http://{args.host}:{args.port}/ "
        f"({args.width}x{args.height}@{args.fps}, quality={args.quality})",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        stop_event.set()
        capture.join(timeout=2.0)
        server.server_close()
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except OSError as exc:
        print(f"error: could not start server on {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
