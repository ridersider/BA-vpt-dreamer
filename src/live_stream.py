import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import cv2
import numpy as np


class _StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress request logs

    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        while True:
            frame = self.server.streamer.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            data = jpeg.tobytes()
            try:
                self.wfile.write(
                    f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n".encode()
                    + data
                    + b"\r\n"
                )
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break

            time.sleep(1 / self.server.streamer.fps)


class LiveStreamer:
    """
    Streams obs['pov'] frames as MJPEG over HTTP.

    Usage:
        streamer = LiveStreamer(port=8080, fps=10)
        streamer.start()
        # inside rollout loop:
        streamer.push_frame(obs["pov"])
    """

    def __init__(self, port: int = 8080, fps: int = 10):
        self.port = port
        self.fps = fps
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._server: Optional[HTTPServer] = None

    def push_frame(self, frame: np.ndarray):
        """Push a new RGB frame (H, W, 3) uint8 — called from the training loop."""
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        with self._lock:
            self._frame = bgr

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def start(self):
        """Start the HTTP server in a daemon thread."""
        server = HTTPServer(("0.0.0.0", self.port), _StreamHandler)
        server.streamer = self
        self._server = server
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"[LiveStreamer] Stream available at http://localhost:{self.port}")
