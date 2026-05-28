"""Flask frontend for the FinStream sentiment demo."""

from __future__ import annotations

import os

import requests
from flask import Flask, Response, abort, render_template, request


def _normalize_api_base_url(raw_value: str | None) -> str:
    """Return a browser-safe API URL for local or Render deployments."""

    value = (raw_value or "http://localhost:8000").strip()
    if not value:
        return "http://localhost:8000"
    if "://" not in value:
        return f"https://{value}"
    return value


def create_app() -> Flask:
    """Create and configure the Flask application."""

    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "finstream-dev-secret")
    app.config["BACKEND_API_URL"] = _normalize_api_base_url(
        os.getenv("BACKEND_API_URL") or os.getenv("FASTAPI_BASE_URL")
    )
    app.config["APP_NAME"] = os.getenv("APP_NAME", "FinStream")
    app.config["REQUEST_TIMEOUT"] = float(os.getenv("BACKEND_REQUEST_TIMEOUT", "120"))

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            api_base_url="/api",
            app_name=app.config["APP_NAME"],
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "frontend"}

    @app.route("/api/<path:path>", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def proxy_api(path: str):
        backend_url = f"{app.config['BACKEND_API_URL'].rstrip('/')}/{path}"

        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }

        files = None
        data = None
        json_data = None
        if request.files:
            files = {
                key: (uploaded.filename, uploaded.stream, uploaded.mimetype)
                for key, uploaded in request.files.items()
            }
            data = request.form.to_dict(flat=True)
        elif request.is_json:
            json_data = request.get_json(silent=True)
        else:
            data = request.form.to_dict(flat=True) or request.get_data()

        try:
            upstream = requests.request(
                method=request.method,
                url=backend_url,
                params=request.args.to_dict(flat=True),
                headers=headers,
                files=files,
                json=json_data,
                data=data if json_data is None and files is None else None,
                timeout=app.config["REQUEST_TIMEOUT"],
            )
        except requests.RequestException as exc:
            abort(502, description=f"Backend request failed: {exc}")

        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in {"content-encoding", "content-length", "transfer-encoding", "connection"}
        }

        return Response(
            upstream.content,
            status=upstream.status_code,
            headers=response_headers,
            content_type=upstream.headers.get("content-type"),
        )

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
