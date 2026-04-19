from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.models.registry import ModelRegistry
from app.providers.csv_provider import CSVTabularProvider

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.project_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def on_startup() -> None:
    registry = ModelRegistry(settings=settings)
    registry.load_all()

    provider = CSVTabularProvider(
        apartment_csv=settings.provider_apartment_csv_path,
        house_csv=settings.provider_house_csv_path,
    )

    app.state.settings = settings
    app.state.registry = registry
    app.state.provider = provider

    logger.info("Application startup completed", extra={"request_id": "startup"})


@app.exception_handler(RequestValidationError)
async def request_validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI validation payload may contain non-JSON objects in ctx.error (e.g. ValueError).
    # jsonable_encoder with a broad Exception encoder keeps the response serializable.
    detail = jsonable_encoder(
        exc.errors(),
        custom_encoder={Exception: lambda value: str(value)},
    )
    return JSONResponse(status_code=422, content={"detail": detail})


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server exception", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
