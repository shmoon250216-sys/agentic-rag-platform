from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import get_settings
from app.core.errors import register_exception_handlers

WEB_DIR = Path(__file__).parent / "web"


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )
    register_exception_handlers(app)
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    app.include_router(router)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/docs")
    async def docs_app() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()
