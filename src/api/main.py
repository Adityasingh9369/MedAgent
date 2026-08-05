from fastapi import FastAPI
from contextlib import asynccontextmanager
from loguru import logger

# from src.api.routes import health
from src.api.routes import health, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 55)
    logger.info("ClinicalAgent API starting up...")
    logger.info("=" * 55)
    yield
    logger.info("ClinicalAgent API shutting down...")


app = FastAPI(
    title="ClinicalAgent API",
    description="Production multi-agent medical intelligence system",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["health"])
app.include_router(query.router, tags=["query"])


@app.get("/")
def root():
    return {"service": "ClinicalAgent API", "status": "running", "docs": "/docs"}