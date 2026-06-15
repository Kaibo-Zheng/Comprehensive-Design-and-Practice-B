"""云台跟踪主程序内置的 Web 监控服务。

与独立摄像头流模块不同，本模块不直接占用摄像头。主跟踪循环负责把最新 JPEG
画面和状态字典推入线程安全的 WebState，HTTP 服务线程只负责读取数据。这样可以
保持摄像头只有一个所有者，同时用一个命令完成跟踪和远程监控。

接口：
    /              HTML 监控页面
    /stream.mjpg   MJPEG 视频流
    /snapshot.jpg  最新单帧 JPEG
    /status.json   JSON 状态数据

Built only on the standard library (``ThreadingHTTPServer``) — no Flask/FastAPI.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional


class WebState:
    """线程安全地保存最新画面和状态。"""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._frame_id = 0
        self._status: Dict[str, Any] = {}
        self.started_at = time.time()

    def update(self, jpeg: Optional[bytes], status: Dict[str, Any]) -> None:
        with self._condition:
            if jpeg is not None:
                self._jpeg = jpeg
                self._frame_id += 1
            self._status = status
            self._condition.notify_all()

    def status(self) -> Dict[str, Any]:
        with self._condition:
            data = dict(self._status)
            data["stream_frames"] = self._frame_id
            data["uptime_s"] = round(time.time() - self.started_at, 2)
            data["has_frame"] = self._jpeg is not None
            return data

    def snapshot(self, timeout: float = 5.0) -> Optional[bytes]:
        with self._condition:
            if self._jpeg is None:
                self._condition.wait(timeout)
            return self._jpeg

    def wait_for_frame(self, last_seen: int, timeout: float = 2.0) -> tuple[int, Optional[bytes]]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._frame_id == last_seen:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return self._frame_id, self._jpeg


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gimbal Tracker Monitor</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #eef3f6;
      --panel: #ffffff;
      --panel-2: #f8fafc;
      --line: #d9e2e8;
      --text: #172026;
      --muted: #647481;
      --soft: #33414a;
      --accent: #0f9f7a;
      --warn: #b7791f;
      --danger: #c2413b;
      --cyan: #2a9fd6;
      --shadow: 0 14px 36px rgba(24, 39, 51, .10);
      --radius: 8px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: linear-gradient(180deg, #f8fbfc 0%, var(--bg) 48%, #e8eef2 100%);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      letter-spacing: 0;
    }
    a { color: inherit; text-decoration: none; }
    .app { height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
    .topbar {
      height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, .92);
      backdrop-filter: blur(12px);
    }
    .brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
    .mark {
      width: 30px;
      height: 30px;
      border: 1px solid #b9d7ce;
      border-radius: 7px;
      display: grid;
      place-items: center;
      color: var(--accent);
      background: #e8f7f2;
      flex: 0 0 auto;
    }
    .mark svg { width: 18px; height: 18px; }
    h1 { margin: 0; font-size: 15px; line-height: 1.2; font-weight: 700; }
    .subtitle { margin-top: 2px; color: var(--muted); font-size: 12px; line-height: 1.2; }
    .actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .link-button {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      height: 32px;
      padding: 0 10px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #ffffff;
      color: var(--soft);
      font-size: 12px;
      font-weight: 650;
    }
    .link-button:hover { border-color: #a9bac4; color: var(--text); box-shadow: 0 6px 16px rgba(24, 39, 51, .08); }
    .link-button svg { width: 15px; height: 15px; }
    .shell {
      flex: 1;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 14px;
      padding: 14px;
      min-height: 0;
      overflow: hidden;
    }
    .video-panel, .side-panel, .tile {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    .video-panel { overflow: hidden; display: flex; flex-direction: column; min-width: 0; min-height: 0; }
    .video-header {
      flex: 0 0 auto;
      min-height: 46px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
    }
    .video-title { display: flex; align-items: center; gap: 9px; min-width: 0; }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--muted);
      box-shadow: 0 0 0 4px rgba(100, 116, 129, .12);
      flex: 0 0 auto;
    }
    .dot.ok { background: var(--accent); box-shadow: 0 0 0 4px rgba(15, 159, 122, .16); }
    .dot.warn { background: var(--warn); box-shadow: 0 0 0 4px rgba(183, 121, 31, .16); }
    .dot.danger { background: var(--danger); box-shadow: 0 0 0 4px rgba(194, 65, 59, .16); }
    .label { color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
    .value { color: var(--text); font-size: 14px; font-weight: 750; white-space: nowrap; }
    .stream-wrap {
      position: relative;
      flex: 1 1 0;
      min-height: 0;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at 50% 50%, rgba(42, 159, 214, .12), transparent 36%),
        #0a0d0f;
      background-size: 32px 32px;
    }
    .stream-wrap::before,
    .stream-wrap::after {
      content: "";
      position: absolute;
      left: 50%;
      top: 50%;
      pointer-events: none;
      background: rgba(42, 159, 214, .5);
      opacity: .45;
    }
    .stream-wrap::before { width: 1px; height: 42px; transform: translate(-.5px, -50%); }
    .stream-wrap::after { width: 42px; height: 1px; transform: translate(-50%, -.5px); }
    .stream {
      width: 100%;
      height: 100%;
      display: block;
      object-fit: contain;
      background: #07090b;
    }
    .video-footer {
      flex: 0 0 auto;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      border-top: 1px solid var(--line);
      background: #f8fafc;
    }
    .metric {
      min-height: 62px;
      padding: 11px 12px;
      border-right: 1px solid var(--line);
      min-width: 0;
    }
    .metric:last-child { border-right: 0; }
    .metric .value { margin-top: 5px; font-size: 18px; }
    .side-panel {
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-width: 0;
      min-height: 0;
      overflow: auto;
    }
    .tile { padding: 12px; background: #ffffff; }
    .tile-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 12px; }
    .tile-title { font-size: 12px; font-weight: 800; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
    .badge {
      height: 24px;
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 0 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--soft);
      background: #f7fafb;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .badge.ok { color: #08785d; border-color: rgba(15, 159, 122, .28); background: #e9f8f3; }
    .badge.warn { color: #8a5a12; border-color: rgba(183, 121, 31, .28); background: #fff7e5; }
    .badge.danger { color: #a3312d; border-color: rgba(194, 65, 59, .28); background: #fff0ef; }
    .stat-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .stat { padding: 10px; border: 1px solid #e0e8ed; border-radius: 7px; background: #f8fafc; min-width: 0; }
    .stat .value { margin-top: 5px; overflow: hidden; text-overflow: ellipsis; }
    .rows { display: grid; gap: 8px; }
    .row { display: flex; align-items: center; justify-content: space-between; gap: 12px; color: var(--muted); font-size: 13px; }
    .row strong { color: var(--text); font-size: 13px; font-weight: 750; text-align: right; overflow-wrap: anywhere; }
    .axis { display: grid; gap: 10px; }
    .bar-row { display: grid; grid-template-columns: 36px 1fr 50px; align-items: center; gap: 10px; color: var(--muted); font-size: 12px; font-weight: 700; }
    .bar { height: 8px; border-radius: 999px; background: #e8eef2; border: 1px solid #d8e2e8; overflow: hidden; }
    .bar-fill { height: 100%; width: 50%; background: var(--cyan); border-radius: inherit; transition: width .25s ease; }
    .target-card {
      display: grid;
      grid-template-columns: 76px 1fr;
      gap: 12px;
      align-items: center;
    }
    .target-box {
      aspect-ratio: 1;
      border: 1px solid #c8ded7;
      border-radius: 7px;
      display: grid;
      place-items: center;
      background:
        linear-gradient(rgba(15,159,122,.12) 1px, transparent 1px),
        linear-gradient(90deg, rgba(15,159,122,.12) 1px, transparent 1px),
        #eefaf6;
      background-size: 14px 14px;
      color: var(--accent);
    }
    .target-box svg { width: 34px; height: 34px; }
    .target-main { min-width: 0; }
    .target-name { font-size: 22px; font-weight: 800; line-height: 1.1; overflow: hidden; text-overflow: ellipsis; }
    .target-sub { margin-top: 6px; color: var(--muted); font-size: 13px; }
    .muted { color: var(--muted); }
    .error { color: var(--danger); }
    .ok-text { color: var(--accent); }
    @media (max-width: 920px) {
      .app { height: auto; min-height: 100vh; overflow: visible; }
      .shell { grid-template-columns: 1fr; height: auto; overflow: visible; }
      .side-panel { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .target-card { grid-column: span 2; }
      .stream-wrap { min-height: 420px; }
    }
    @media (max-width: 640px) {
      .topbar { height: auto; align-items: flex-start; padding: 12px; flex-direction: column; }
      .actions { width: 100%; justify-content: stretch; }
      .link-button { flex: 1 1 auto; justify-content: center; }
      .shell { padding: 10px; gap: 10px; }
      .stream-wrap { min-height: 240px; }
      .video-footer { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metric:nth-child(2) { border-right: 0; }
      .metric:nth-child(-n+2) { border-bottom: 1px solid var(--line); }
      .side-panel { display: flex; }
      .target-card { grid-template-columns: 60px 1fr; }
      .target-name { font-size: 19px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="brand">
        <div class="mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <path d="M12 3v4M12 17v4M3 12h4M17 12h4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
            <circle cx="12" cy="12" r="5" stroke="currentColor" stroke-width="1.8"/>
            <circle cx="12" cy="12" r="1.4" fill="currentColor"/>
          </svg>
        </div>
        <div>
          <h1>Gimbal Tracker Monitor</h1>
          <div class="subtitle">Live video, target lock, and servo telemetry</div>
        </div>
      </div>
      <nav class="actions" aria-label="Monitor links">
        <a class="link-button" href="/snapshot.jpg">
          <svg viewBox="0 0 24 24" fill="none"><path d="M4 7h3l1.5-2h7L17 7h3v12H4V7Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><circle cx="12" cy="13" r="3.2" stroke="currentColor" stroke-width="1.8"/></svg>
          Snapshot
        </a>
        <a class="link-button" href="/status.json">
          <svg viewBox="0 0 24 24" fill="none"><path d="M8 7h8M8 12h8M8 17h5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><rect x="5" y="3" width="14" height="18" rx="2" stroke="currentColor" stroke-width="1.8"/></svg>
          JSON
        </a>
      </nav>
    </header>
    <main class="shell">
      <section class="video-panel" aria-label="Live stream">
        <div class="video-header">
          <div class="video-title">
            <span id="stream-dot" class="dot"></span>
            <div>
              <div class="label">Live stream</div>
              <div id="stream-state" class="value">Waiting for frame</div>
            </div>
          </div>
          <div class="badge" id="detector-badge">detector --</div>
        </div>
        <div class="video-footer">
          <div class="metric"><div class="label">FPS</div><div id="fps" class="value">--</div></div>
          <div class="metric"><div class="label">Error</div><div id="error" class="value">--</div></div>
          <div class="metric"><div class="label">Frame</div><div id="frame" class="value">--</div></div>
          <div class="metric"><div class="label">Uptime</div><div id="uptime" class="value">--</div></div>
        </div>
        <div class="stream-wrap">
          <img class="stream" src="/stream.mjpg" alt="Live tracking stream">
        </div>
      </section>
      <aside class="side-panel" aria-label="Telemetry">
        <section class="tile target-card">
          <div id="target-icon" class="target-box" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none"><path d="M5 12h14M12 5v14" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/><circle cx="12" cy="12" r="6" stroke="currentColor" stroke-width="1.7"/></svg>
          </div>
          <div class="target-main">
            <div id="target-name" class="target-name">No target</div>
            <div id="target-sub" class="target-sub">Waiting for detector update</div>
          </div>
        </section>
        <section class="tile">
          <div class="tile-head">
            <div class="tile-title">Servo</div>
            <span id="servo-badge" class="badge">pan / tilt</span>
          </div>
          <div class="axis">
            <div class="bar-row"><span>Pan</span><div class="bar"><div id="pan-bar" class="bar-fill"></div></div><strong id="pan">--</strong></div>
            <div class="bar-row"><span>Tilt</span><div class="bar"><div id="tilt-bar" class="bar-fill"></div></div><strong id="tilt">--</strong></div>
          </div>
        </section>
        <section class="tile">
          <div class="tile-head">
            <div class="tile-title">Performance</div>
            <span id="perf-badge" class="badge">timing</span>
          </div>
          <div class="stat-grid">
            <div class="stat"><div class="label">Capture</div><div id="capture-ms" class="value">--</div></div>
            <div class="stat"><div class="label">Infer</div><div id="infer-ms" class="value">--</div></div>
            <div class="stat"><div class="label">Post</div><div id="post-ms" class="value">--</div></div>
            <div class="stat"><div class="label">Total</div><div id="total-ms" class="value">--</div></div>
          </div>
        </section>
        <section class="tile">
          <div class="tile-head">
            <div class="tile-title">Status</div>
            <span id="lock-badge" class="badge">unknown</span>
          </div>
          <div class="rows">
            <div class="row"><span>Confidence</span><strong id="confidence">--</strong></div>
            <div class="row"><span>Lost frames</span><strong id="lost">--</strong></div>
            <div class="row"><span>Stream frames</span><strong id="stream-frames">--</strong></div>
            <div class="row"><span>Offset</span><strong id="offset">--</strong></div>
          </div>
        </section>
      </aside>
    </main>
  </div>
  <script>
    const $ = (id) => document.getElementById(id);
    function num(value, digits = 1) {
      const n = Number(value);
      return Number.isFinite(n) ? n.toFixed(digits) : '--';
    }
    function ms(value) {
      const n = Number(value);
      return Number.isFinite(n) ? n.toFixed(1) + ' ms' : '--';
    }
    function pctFromAngle(value, low, high) {
      const n = Number(value);
      if (!Number.isFinite(n)) return 50;
      return Math.max(0, Math.min(100, ((n - low) / (high - low)) * 100));
    }
    function setBadge(el, text, state) {
      el.textContent = text;
      el.className = 'badge' + (state ? ' ' + state : '');
    }
    function setDot(el, state) {
      el.className = 'dot' + (state ? ' ' + state : '');
    }
    async function poll() {
      try {
        const r = await fetch('/status.json', {cache: 'no-store'});
        const s = await r.json();
        const found = Boolean(s.target_found);
        const stale = Boolean(s.target_stale);
        const hasFrame = Boolean(s.has_frame);
        const detector = s.detector || '--';
        const lockState = found ? (stale ? 'warn' : 'ok') : 'danger';
        setDot($('stream-dot'), hasFrame ? 'ok' : 'warn');
        $('stream-state').textContent = hasFrame ? 'Receiving frames' : 'Waiting for frame';
        setBadge($('detector-badge'), detector, detector.includes('rknn') ? 'ok' : '');
        $('fps').textContent = num(s.fps, 1);
        $('error').textContent = Number.isFinite(Number(s.error_norm)) ? Math.round(Number(s.error_norm)) + ' px' : '--';
        $('frame').textContent = s.frame_index ?? '--';
        $('uptime').textContent = Number.isFinite(Number(s.uptime_s)) ? Math.round(Number(s.uptime_s)) + ' s' : '--';
        $('target-name').textContent = found ? (s.target_label || 'target') : 'No target';
        $('target-sub').textContent = found
          ? (stale ? 'Holding last known target' : 'Target locked')
          : 'Detector has not selected a target';
        $('target-icon').style.color = found ? (stale ? 'var(--warn)' : 'var(--accent)') : 'var(--danger)';
        $('pan').textContent = num(s.pan, 1) + '°';
        $('tilt').textContent = num(s.tilt, 1) + '°';
        $('pan-bar').style.width = pctFromAngle(s.pan, 45, 135) + '%';
        $('tilt-bar').style.width = pctFromAngle(s.tilt, 60, 120) + '%';
        setBadge($('servo-badge'), num(s.pan, 0) + '° / ' + num(s.tilt, 0) + '°', '');
        $('capture-ms').textContent = ms(s.capture_ms);
        $('infer-ms').textContent = ms(s.inference_ms);
        $('post-ms').textContent = ms(s.postprocess_ms);
        $('total-ms').textContent = ms(s.total_ms);
        setBadge($('perf-badge'), num(s.total_ms, 0) + ' ms', Number(s.total_ms) > 120 ? 'warn' : 'ok');
        setBadge($('lock-badge'), found ? (stale ? 'stale' : 'locked') : 'lost', lockState);
        $('confidence').textContent = s.target_confidence == null ? '--' : Number(s.target_confidence).toFixed(3);
        $('lost').textContent = s.lost_frames ?? '--';
        $('stream-frames').textContent = s.stream_frames ?? '--';
        $('offset').textContent = '(' + (s.x_error ?? 0) + ', ' + (s.y_error ?? 0) + ') px';
      } catch (e) {
        setDot($('stream-dot'), 'danger');
        $('stream-state').textContent = 'Status API unavailable';
        setBadge($('lock-badge'), 'offline', 'danger');
      }
      setTimeout(poll, 500);
    }
    poll();
  </script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: "MonitorServer"

    def log_message(self, *_args: Any) -> None:  # 关闭每个请求的默认日志。
        pass

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send_bytes(_INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/status.json":
            body = json.dumps(self.server.state.status(), ensure_ascii=False, indent=2).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8")
        elif path == "/snapshot.jpg":
            self._send_snapshot()
        elif path == "/stream.mjpg":
            self._send_stream()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_HEAD(self) -> None:  # noqa: N802
        """支持对数据接口执行只返回响应头的 HEAD 请求。"""
        path = self.path.split("?", 1)[0]
        content_types = {
            "/": "text/html; charset=utf-8",
            "/status.json": "application/json; charset=utf-8",
            "/snapshot.jpg": "image/jpeg",
            "/stream.mjpg": "multipart/x-mixed-replace; boundary=FRAME",
        }
        if path not in content_types:
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        if path == "/snapshot.jpg" and self.server.state.snapshot(timeout=0.0) is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "no frame yet")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_types[path])
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_snapshot(self) -> None:
        jpeg = self.server.state.snapshot()
        if jpeg is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "no frame yet")
            return
        self._send_bytes(jpeg, "image/jpeg")

    def _send_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()
        last_seen = 0
        try:
            while not self.server.stop_event.is_set():
                frame_id, jpeg = self.server.state.wait_for_frame(last_seen)
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


class MonitorServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], state: WebState, stop_event: threading.Event) -> None:
        super().__init__(server_address, _Handler)
        self.state = state
        self.stop_event = stop_event


class WebMonitor:
    """持有 HTTP 服务线程，由主循环通过 update 推送数据。"""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.state = WebState()
        self.stop_event = threading.Event()
        self._server: Optional[MonitorServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._server = MonitorServer((self.host, self.port), self.state, self.stop_event)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        kwargs={"poll_interval": 0.2}, daemon=True)
        self._thread.start()

    def update(self, jpeg: Optional[bytes], status: Dict[str, Any]) -> None:
        self.state.update(jpeg, status)

    def stop(self) -> None:
        self.stop_event.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
