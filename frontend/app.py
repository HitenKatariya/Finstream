"""Flask frontend for the FinStream sentiment demo."""

from __future__ import annotations

import os

from flask import Flask, render_template


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
    app.config["FASTAPI_BASE_URL"] = _normalize_api_base_url(os.getenv("FASTAPI_BASE_URL"))
    app.config["APP_NAME"] = os.getenv("APP_NAME", "FinStream")

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            api_base_url=app.config["FASTAPI_BASE_URL"],
            app_name=app.config["APP_NAME"],
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "frontend"}

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
