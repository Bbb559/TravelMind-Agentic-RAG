"""只读托管 frontend/dist，并将 /api 代理到本地 FastAPI。"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "frontend" / "dist"


class FrontendSmokeHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, backend_url: str, **kwargs) -> None:
        self.backend_url = backend_url.rstrip("/")
        super().__init__(*args, directory=str(DIST), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/"):
            self._proxy()
            return
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path.startswith("/api/"):
            self._proxy()
            return
        self.send_error(405)

    def log_message(self, format: str, *args) -> None:
        return

    def _proxy(self) -> None:
        content_length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(content_length) if content_length else None
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        if run_id := self.headers.get("X-TravelMind-Run-Id"):
            headers["X-TravelMind-Run-Id"] = run_id
        request = urllib.request.Request(
            f"{self.backend_url}{self.path}",
            data=body,
            headers=headers,
            method=self.command,
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
                payload = response.read()
                self.send_response(response.status)
                self.send_header(
                    "Content-Type",
                    response.headers.get("Content-Type", "application/json"),
                )
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception:
            payload = json.dumps({"detail": "proxy_unavailable"}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5173)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    def handler(*handler_args, **handler_kwargs):
        return FrontendSmokeHandler(
            *handler_args,
            backend_url=args.backend_url,
            **handler_kwargs,
        )

    server = ThreadingHTTPServer((args.host, args.port), handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
