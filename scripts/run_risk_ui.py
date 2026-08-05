"""Serve the basic pre-trade risk interface on localhost only."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv

from commodity_ml.intraday.risk_ui import (
    RiskUIError,
    evaluate_read_only_request,
    render_request_report,
)


MAX_REQUEST_BYTES = 64 * 1024
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RiskUIHandler(BaseHTTPRequestHandler):
    server_version = "CommodityRiskUI/1"

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/health":
            self._json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "mode": "OANDA_V20_READ_ONLY",
                    "order_authority": False,
                },
            )
            return
        if path.startswith("/api/"):
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._static(path)

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        try:
            document = self._request_json()
            if path == "/api/evaluate":
                result = evaluate_read_only_request(
                    document,
                    project_root=self.server.project_root,
                )
                self._json(HTTPStatus.OK, result)
            elif path == "/api/report-html":
                html = render_request_report(document).encode("utf-8")
                self._send(HTTPStatus.OK, html, "text/html; charset=utf-8")
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (RiskUIError, ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _request_json(self) -> dict[str, Any]:
        if self.headers.get_content_type() != "application/json":
            raise RiskUIError("Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise RiskUIError("invalid Content-Length") from exc
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise RiskUIError("request body is empty or too large")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise RiskUIError("request body must be a JSON object")
        return value

    def _static(self, request_path: str) -> None:
        relative = "index.html" if request_path in ("", "/") else request_path.lstrip("/")
        root = self.server.static_root.resolve()
        candidate = (root / relative).resolve()
        if root not in candidate.parents and candidate != root:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not candidate.is_file():
            candidate = root / "index.html"
        if not candidate.is_file():
            self._json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "UI build missing; run npm install and npm run build in web"},
            )
            return
        media_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self._send(HTTPStatus.OK, candidate.read_bytes(), media_type)

    def _json(self, status: HTTPStatus, value: Any) -> None:
        body = json.dumps(value, allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"Risk UI: {format % args}")


class RiskUIServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], static_root: Path, project_root: Path):
        super().__init__(address, RiskUIHandler)
        self.static_root = static_root
        self.project_root = project_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--static-root", type=Path, default=PROJECT_ROOT / "web/dist")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    server = RiskUIServer((args.host, args.port), args.static_root, PROJECT_ROOT)
    print(f"Risk UI -> http://{args.host}:{args.port}")
    print("Read-only decision support: no order endpoint exists.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
