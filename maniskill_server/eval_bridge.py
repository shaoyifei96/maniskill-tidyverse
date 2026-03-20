"""Evaluation bridge — exposes task evaluate() and task_info via ZMQ REP socket.

Protocol: JSON REQ/REP on port 5590
  REQ: {"command": "evaluate"} → REP: {"success": true/false, ...}
  REQ: {"command": "task_info"} → REP: {"task_id": ..., "task_class": ..., ...}
  REQ: {"command": "reset", "seed": 42} → REP: {"ok": true}
"""

import json
import threading
import zmq

EVAL_PORT = 5590


class EvalBridge:
    """ZMQ REP bridge for task evaluation queries."""

    def __init__(self, server, port=EVAL_PORT):
        self.server = server
        self.port = port
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="eval-bridge")
        self._thread.start()
        print(f"[eval-bridge] Listening on tcp://*:{self.port}")

    def stop(self):
        self._running = False

    def _run(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout for clean shutdown
        sock.bind(f"tcp://*:{self.port}")

        while self._running:
            try:
                msg = sock.recv_json()
            except zmq.Again:
                continue
            except Exception as e:
                print(f"[eval-bridge] recv error: {e}")
                continue

            cmd = msg.get("command", "")
            try:
                if cmd == "evaluate":
                    result = self.server.submit_command("evaluate")
                    sock.send_json(result)
                elif cmd == "task_info":
                    result = self.server.submit_command("get_task_info")
                    sock.send_json(result)
                elif cmd == "reset":
                    seed = msg.get("seed", None)
                    self.server.submit_command("reset", seed=seed)
                    sock.send_json({"ok": True})
                else:
                    sock.send_json({"error": f"unknown command: {cmd}"})
            except Exception as e:
                try:
                    sock.send_json({"error": str(e)})
                except Exception:
                    pass

        sock.close()
        ctx.term()
