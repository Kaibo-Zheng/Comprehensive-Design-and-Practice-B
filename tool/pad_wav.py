#!/usr/bin/env python3
"""Add leading/trailing silence to a WAV file."""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pad a WAV with silence")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--lead-ms", type=int, default=500)
    parser.add_argument("--trail-ms", type=int, default=100)
    parser.add_argument("--lead-tone-ms", type=int, default=0)
    parser.add_argument("--trail-tone-ms", type=int, default=0)
    parser.add_argument("--tone-hz", type=float, default=440.0)
    parser.add_argument("--tone-level", type=float, default=0.03)
    return parser.parse_args()


def tone_bytes(params: wave._wave_params, duration_ms: int, hz: float, level: float) -> bytes:
    if duration_ms <= 0:
        return b""
    if params.sampwidth != 2:
        raise ValueError("tone generation currently supports 16-bit PCM WAV only")
    frames = max(0, round(params.framerate * duration_ms / 1000))
    amplitude = max(0, min(32767, int(32767 * level)))
    out = bytearray()
    for frame_index in range(frames):
        sample = int(amplitude * math.sin(2.0 * math.pi * hz * frame_index / params.framerate))
        encoded = sample.to_bytes(2, byteorder="little", signed=True)
        out.extend(encoded * params.nchannels)
    return bytes(out)


def main() -> int:
    args = parse_args()
    source = Path(args.input).expanduser()
    output = Path(args.output).expanduser()
    with wave.open(str(source), "rb") as src:
        params = src.getparams()
        frames = src.readframes(src.getnframes())
    frame_bytes = params.nchannels * params.sampwidth
    lead_frames = max(0, round(params.framerate * args.lead_ms / 1000))
    trail_frames = max(0, round(params.framerate * args.trail_ms / 1000))
    silence = b"\x00" * frame_bytes
    lead_tone = tone_bytes(params, args.lead_tone_ms, args.tone_hz, args.tone_level)
    trail_tone = tone_bytes(params, args.trail_tone_ms, args.tone_hz, args.tone_level)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as dst:
        dst.setparams(params)
        dst.writeframes(lead_tone + silence * lead_frames + frames + silence * trail_frames + trail_tone)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
