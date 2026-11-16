import json
import logging
from pathlib import Path
from typing import Any

from .dataset import DatasetManager
from .agent import build_agent
from .tools import create_tools

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
        row_count = len(self.dataset_manager.get_dataframe())
        logger.info("Loaded %d rows from %s", row_count, resolved.name)
        return {"status": "loaded", "rows": row_count}

    def initialize_agent(self) -> dict:
        if self.dataset_manager.get_dataframe() is None:
            raise RuntimeError("No dataset loaded. Call load_dataset() first.")
            
        self.agent = build_agent(self.dataset_manager)
        logger.info("Agent initialized successfully")
        return {"status": "ready"}

    def ask(self, query: str) -> dict:
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call initialize_agent() first.")
        
        response = self.agent.invoke({"messages": [{"role": "user", "content": query}]})
        content = response["messages"][-1].content
        logger.info("Query answered: %s", query[:80])
        return {"answer": content}

    def clean_dataset(self, strategy: str, columns: list[str] | None = None) -> dict:
        """Execute data cleaning via the registered tool."""
        if self.dataset_manager.get_dataframe() is None:
            raise RuntimeError("No dataset loaded.")

        tools = create_tools(self.dataset_manager)
        clean_tool = next((t for t in tools if t.name == "clean_dataset"), None)
        
        if clean_tool is None:
            raise ValueError("clean_dataset tool not available")

        result_str = clean_tool.invoke({
            "strategy": strategy, 
            "columns": columns or []
        })
        
        # Parse JSON string returned by tool into structured dict
        try:
            return json.loads(result_str) if isinstance(result_str, str) else result_str
        except json.JSONDecodeError:
            return {"raw_result": str(result_str)}