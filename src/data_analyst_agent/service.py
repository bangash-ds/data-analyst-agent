import logging
from pathlib import Path
from typing import Any

from .dataset import DatasetManager
from .agent import build_agent

logger = logging.getLogger(__name__)


class AnalystService:
    """Reusable agent orchestration independent of any UI."""

    def __init__(self):
        self.dataset_manager = DatasetManager()
        self.agent: Any | None = None

    def load_dataset(self, path: str) -> dict:
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset not found: {resolved}")
        self.dataset_manager.load_dataset(str(resolved))
        logger.info("Loaded %d rows", len(self.dataset_manager.get_dataframe()))
        return {"status": "loaded", "rows": len(self.dataset_manager.get_dataframe())}

    def initialize_agent(self) -> dict:
        self.agent = build_agent(self.dataset_manager)
        logger.info("Agent initialized")
        return {"status": "ready"}

    def ask(self, query: str) -> dict:
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call initialize_agent() first.")
        response = self.agent.invoke({"messages": [{"role": "user", "content": query}]})
        content = response["messages"][-1].content
        logger.info("Query answered: %s", query[:80])
        return {"answer": content}