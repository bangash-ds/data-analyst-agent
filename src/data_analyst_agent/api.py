import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse  # ✅ Added for root redirect
from pydantic import BaseModel, Field

from .service import AnalystService
from .schemas import CleaningRequest

logger = logging.getLogger(__name__)

_service: AnalystService | None = None


class DatasetLoadRequest(BaseModel):
    path: str = Field(..., description="Absolute or relative path to CSV dataset")


class QueryRequest(BaseModel):
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=5000, 
        description="Natural language analysis question"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize service on startup; cleanup on shutdown."""
    global _service
    logger.info("Starting AI Data Analyst API...")
    _service = AnalystService()
    yield
    logger.info("Shutting down AI Data Analyst API...")
    _service = None


app = FastAPI(
    title="AI Data Analyst Agent API",
    version="1.0.0",
    description="Production REST API for exploratory data analysis via LangChain agent.",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)  # ✅ Permanent fix: Auto-redirect to docs
async def root():
    """Redirect root URL to Swagger UI documentation."""
    return RedirectResponse(url="/docs")


@app.post("/load-dataset", status_code=200)
async def load_dataset(request: DatasetLoadRequest) -> dict[str, Any]:
    """Load a dataset into the agent's memory."""
    if _service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        return _service.load_dataset(request.path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Dataset load failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post("/initialize-agent", status_code=200)
async def initialize_agent() -> dict[str, str]:
    """Build and initialize the LangChain ReAct agent."""
    if _service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        return _service.initialize_agent()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error("Agent init failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Agent initialization failed: {e}")


@app.post("/ask", status_code=200)
async def ask_question(request: QueryRequest) -> dict[str, str]:
    """Submit a natural language question to the analyst agent."""
    if _service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        return _service.ask(request.query)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error("Query execution failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.post("/clean-dataset", status_code=200)
async def clean_dataset(request: CleaningRequest) -> dict[str, Any]:
    """Execute a data cleaning strategy and save cleaned CSV."""
    if _service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        return _service.clean_dataset(
            strategy=request.strategy, 
            columns=request.columns
        )
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error("Cleaning failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Cleaning failed: {e}")