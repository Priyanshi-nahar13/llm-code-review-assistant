"""
src/api/server.py

FastAPI server that receives GitHub webhooks and returns review findings.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from src.model.inference import get_engine
from src.parser.diff_parser import parse_diff
from .routes import router


# ─── Lifespan (model pre-load) ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Pre-loading model on startup …")
    try:
        get_engine()._ensure_loaded()
        logger.info("Model loaded ✓")
    except Exception as e:
        logger.warning(f"Model pre-load failed (will retry on first request): {e}")
    yield
    logger.info("Shutting down")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LLM Code Review Assistant",
    description="Automated first-pass PR reviews powered by fine-tuned CodeLlama",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# ─── Health ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("MODEL_PATH", "not configured")}
