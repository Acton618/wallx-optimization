#!/usr/bin/env python3
"""Simulate the Quantum-1 robot TCP client for testing the bridge.

Follows the legacy protocol from robot_controller.py:
    robot -> bridge : [u32 len][state json], then 3x [u32 len][jpeg bytes]
                      (camera_left, camera_front, camera_right)
"""
import socket
import struct
import json
import numpy as np
import cv2


def send_frame(sock, data):
    sock.sendall(struct.pack("<L", len(data)))
    sock.sendall(data)


def main():
    sock = socket.create_connection(("127.0.0.1", 30123), timeout=15)
    state = {"follow1_pos": [0.0] * 7, "follow2_pos": [0.0] * 7}
    send_frame(sock, json.dumps(state).encode("utf-8"))
    img = np.zeros((448, 448, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    for i in range(3):
        send_frame(sock, buf.tobytes())
    print("sent state + 3 images", flush=True)

    sock.settimeout(8)
    try:
        size_b = sock.recv(4)
        if not size_b:
            print("connection closed by bridge (dry-run expected)", flush=True)
            return
        size = struct.unpack("<L", size_b)[0]
        data = b""
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                break
            data += chunk
        print("received action json:", data.decode("utf-8")[:300], flush=True)
    except socket.timeout:
        print("timeout waiting for action (dry-run expected)", flush=True)
    except Exception as e:
        print("recv error:", e, flush=True)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
