import logging
from typing import Any

from langchain.agents import create_agent
from langchain_groq import ChatGroq

from .config import GROQ_API_KEY, GROQ_MODEL
from .tools import create_tools
from .prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def build_agent(dataset_manager) -> Any:
    """
    Builds and returns a LangChain ReAct agent.

    Args:
        dataset_manager: An instance of DatasetManager.

    Returns:
        A compiled LangChain agent executor.
    """
    logger.info("Initializing LLM with model: %s", GROQ_MODEL)

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )

    tools = create_tools(dataset_manager)
    logger.info("Registered %d tools with the agent", len(tools))

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    logger.info("Agent built successfully")
    return agent