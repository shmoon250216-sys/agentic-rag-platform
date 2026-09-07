from fastapi import Header

from app.core.config import get_settings
from app.core.errors import AppError


async def verify_token(authorization: str | None = Header(default=None)) -> None:
    settings = get_settings()
    expected = f"Bearer {settings.api_token}"
    if authorization != expected:
        raise AppError(
            code="UNAUTHORIZED",
            message="Missing or invalid API token",
            status_code=401,
        )

