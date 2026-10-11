#!/usr/bin/env python3
"""TCP <-> WebSocket bridge: legacy Quantum-1 robot TCP protocol to
Wall-OSS-0.5 official WebSocket serving.

Architecture:
    Quantum-1 robot (legacy TCP client, e.g. `infer ip port`)
        <-> legacy TCP on this bridge (default 30123)
    this bridge
        <-> Wall-OSS-0.5 WebSocket serving (default ws://127.0.0.1:32195)

Legacy TCP protocol (from robot_controller.py):
    robot -> bridge : [u32 len][state json], then 3x [u32 len][jpeg bytes]
                      (camera_left, camera_front, camera_right)
    bridge -> robot : [u32 len][action json dict]

Safety: default DRY-RUN. It converts and runs inference but does NOT send
actions back to the robot. Physical sending is gated by BOTH --allow-send
and ACTION_SEMANTICS_CONFIRMED (source-code flag) so a CLI typo alone cannot
enable motion.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import struct
import sys

import numpy as np
import cv2
import msgpack
import msgpack_numpy as _m
import websockets

_m.patch()

# Legacy wire order of the three cameras sent by the robot.
LEGACY_CAM_ORDER = ("camera_left", "camera_front", "camera_right")
# Cameras the Wall-OSS-0.5 server requires (2 of the 3).
SERVE_FRONT_KEY = "camera_front"
SERVE_RIGHT_KEY = "camera_right"

# Keep False until the arm action semantics (absolute vs relative) and the
# camera wire order have been confirmed against a real robot capture.
ACTION_SEMANTICS_CONFIRMED = False

logger = logging.getLogger("bridge")


async def _recvall(reader, n):
    buf = b""
    while len(buf) < n:
        chunk = await reader.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


async def _recv_frame(reader):
    size_b = await _recvall(reader, 4)
    if size_b is None:
        return None
    size = struct.unpack("<L", size_b)[0]
    return await _recvall(reader, size)


async def _recv_state(reader):
    raw = await _recv_frame(reader)
    if raw is None:
        return None
    state = json.loads(raw.decode("utf-8"))
    if not isinstance(state, dict):
        raise ValueError("robot state is not a JSON object")
    for key in ("follow1_pos", "follow2_pos"):
        if key not in state:
            raise ValueError(f"robot state missing required key {key!r}")
    return state


async def _recv_image(reader, index):
    raw = await _recv_frame(reader)
    if raw is None:
        raise ConnectionError("robot closed during image stream")
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"failed to decode image #{index}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _img_to_b64(rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("failed to JPEG-encode camera image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _as_action_chunk(value, key):
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(f"{key} must have shape (T,7), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{key} contains NaN/Inf")
    return arr


def prepare_robot_actions(ws_action):
    """Map Wall-OSS-0.5 serialized arms to the legacy robot action dict."""
    if not isinstance(ws_action, dict):
        raise ValueError("WebSocket action response is not a dict")
    if "follow1_pos" not in ws_action or "follow2_pos" not in ws_action:
        raise ValueError("WebSocket response missing follow1_pos/follow2_pos")
    left = _as_action_chunk(ws_action["follow1_pos"], "follow1_pos")
    right = _as_action_chunk(ws_action["follow2_pos"], "follow2_pos")
    if left.shape[0] != right.shape[0]:
        raise ValueError("follow1_pos/follow2_pos trajectory length mismatch")
    length = int(left.shape[0])
    return {
        "follow1_pos": left.tolist(),
        "follow2_pos": right.tolist(),
        "head_pos": [[0.0, -1.0] for _ in range(length)],
        "lift": [0.4 for _ in range(length)],
        "car_pose": [[0.0, 0.0, 0.0] for _ in range(length)],
    }


async def _handle_client(reader, writer, ws_url, instruction, allow_send):
    addr = writer.get_extra_info("peername")
    logger.info("Robot connected: %s", addr)

    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.recv()  # consume server metadata

        # One observation per connection in dry-run; continuous loop when sending.
        while True:
            state = await _recv_state(reader)
            if state is None:
                logger.info("Robot closed the connection")
                return

            images = [await _recv_image(reader, i) for i in range(len(LEGACY_CAM_ORDER))]
            left_img, front_img, right_img = images
            logger.info(
                "recv state keys=%s images=%sx%s",
                list(state.keys()),
                front_img.shape[:2],
                right_img.shape[:2],
            )

            obs = {
                "state": dict(state),
                "views": {
                    SERVE_FRONT_KEY: _img_to_b64(front_img),
                    SERVE_RIGHT_KEY: _img_to_b64(right_img),
                },
                "instruction": instruction,
                "infer_mode": "flow",
            }
            logger.info("forwarding to serving: %s", ws_url)
            await ws.send(msgpack.packb(obs))
            resp = msgpack.unpackb(await ws.recv())

            actions = prepare_robot_actions(resp)
            logger.info(
                "predicted follow1 T=%d last=%s", len(actions["follow1_pos"]), actions["follow1_pos"][-1]
            )
            logger.info(
                "predicted follow2 T=%d last=%s", len(actions["follow2_pos"]), actions["follow2_pos"][-1]
            )

            if not allow_send:
                logger.warning("DRY-RUN: action NOT sent to robot")
                return
            if not ACTION_SEMANTICS_CONFIRMED:
                logger.error("ACTION_SEMANTICS_CONFIRMED=False; refusing to send to robot")
                return

            payload = json.dumps(actions).encode("utf-8")
            writer.write(struct.pack("<L", len(payload)))
            writer.write(payload)
            await writer.drain()
            logger.info("sent action back to robot T=%d", len(actions["follow1_pos"]))


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcp-host", default="0.0.0.0")
    parser.add_argument("--tcp-port", type=int, default=30123)
    parser.add_argument("--ws-url", default="ws://127.0.0.1:32195")
    parser.add_argument("--instruction", default="pick up the object")
    parser.add_argument(
        "--allow-send",
        action="store_true",
        help="Request sending actions back to robot (still gated by ACTION_SEMANTICS_CONFIRMED).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    async def client_connected(reader, writer):
        try:
            await _handle_client(reader, writer, args.ws_url, args.instruction, args.allow_send)
        except Exception as exc:
            logger.error("handler error: %s: %s", type(exc).__name__, exc)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.info("robot session closed")

    server = await asyncio.start_server(client_connected, args.tcp_host, args.tcp_port)
    logger.info(
        "bridge listening on %s:%s -> %s (allow_send=%s)",
        args.tcp_host, args.tcp_port, args.ws_url, args.allow_send,
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
