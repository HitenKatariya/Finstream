"""Flask frontend for the FinStream sentiment demo."""

from __future__ import annotations

import os

import requests
from flask import Flask, Response, abort, render_template, request


def _normalize_api_base_url(raw_value: str | None) -> str:
    value = (raw_value or "").strip()
    if not value:
        return ""
    if "://" not in value:
        return f"http://{value}"
    return value


BACKEND_FALLBACK_URL = _normalize_api_base_url(os.getenv("BACKEND_FALLBACK_URL"))


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "finstream-dev-secret")
    primary_url = os.getenv("BACKEND_API_URL") or os.getenv("FASTAPI_BASE_URL") or ""
    app.config["BACKEND_API_URL"] = _normalize_api_base_url(primary_url)
    if not primary_url:
        app.config["BACKEND_API_URL"] = "http://localhost:8000"
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

    def _try_request(url: str, method: str, headers: dict, files, json_data, data, params, timeout: float):
        return requests.request(
            method=method,
            url=url,
            params=params,
            headers=headers,
            files=files,
            json=json_data,
            data=data if json_data is None and files is None else None,
            timeout=timeout,
        )

    def _build_response_headers(upstream):
        return {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in {"content-encoding", "content-length", "transfer-encoding", "connection"}
        }

    @app.route("/api/<path:path>", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def proxy_api(path: str):
        primary_url = f"{app.config['BACKEND_API_URL'].rstrip('/')}/{path}"

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

        urls_to_try = [primary_url]
        if BACKEND_FALLBACK_URL:
            urls_to_try.append(f"{BACKEND_FALLBACK_URL.rstrip('/')}/{path}")

        last_error = None
        for url in urls_to_try:
            try:
                upstream = _try_request(
                    url, request.method, headers, files, json_data, data,
                    request.args.to_dict(flat=True), app.config["REQUEST_TIMEOUT"],
                )
                if upstream.status_code < 500:
                    return Response(
                        upstream.content,
                        status=upstream.status_code,
                        headers=_build_response_headers(upstream),
                        content_type=upstream.headers.get("content-type"),
                    )
                last_error = f"Backend returned {upstream.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)
                continue

        abort(502, description=f"All backends failed: {last_error}")

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
