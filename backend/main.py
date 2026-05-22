"""
NeuroCalc - Lightweight Handwritten Math Solver
CPU-only, Windows-compatible, offline-first
"""

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from api.routes import router
from database.db import init_db

app = FastAPI(
    title="NeuroCalc",
    description="Lightweight CPU-only Handwritten Math Solver",
    version="1.0.0",
    docs_url="/api/docs",
)

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Serve frontend
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))

@app.on_event("startup")
async def startup_event():
    init_db()
    print("✓ NeuroCalc started — CPU-only mode")
    print("✓ Visit http://localhost:8000")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
