$ErrorActionPreference = "Stop"

uvicorn app.main:create_app --factory --reload

