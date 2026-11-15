# 10. Main Application
# Now we will connect all the components.

# This file will:
# 1. Ask the user for a dataset path.
# 2. Load the dataset.
# 3. Initialize the LangChain agent.
# 4. Accept questions from the user.
# 5. Execute the appropriate analytical tools.
# 6. Display the final answer.


import logging
import sys
from pathlib import Path

from .dataset import DatasetManager
from .agent import build_agent

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the AI Data Analyst Agent CLI."""

    print("\n====================================")
    print("       AI DATA ANALYST AGENT")
    print("====================================\n")

    # Step 1: Initialize dataset manager
    logger.info("Initializing DatasetManager")
    dataset_manager = DatasetManager()

    # Step 2: Ask for dataset path with validation
    while True:
        raw_input = input("Enter your dataset path (or 'exit' to quit): ").strip()
        dataset_path = raw_input.strip('\"\'()[]{}').strip()
        if dataset_path.lower() in ("exit", "quit"):
            print("\nGoodbye!")
            return

        if not dataset_path:
            print("Error: Path cannot be empty. Please try again.")
            continue

        resolved_path = Path(dataset_path).resolve()
        if not resolved_path.exists():
            print(f"Error: File not found at '{resolved_path}'. Please try again.")
            continue

        break

    # Step 3: Load dataset
    logger.info("Loading dataset from: %s", resolved_path)
    try:
        dataset_manager.load_dataset(str(resolved_path))
        logger.info("Dataset loaded successfully: %d rows", len(dataset_manager.get_dataframe()))
    except Exception as error:
        logger.error("Dataset loading failed: %s", error)
        print(f"\nDataset loading failed: {error}")
        return

    # Step 4: Build the agent
    logger.info("Building AI Data Analyst Agent...")
    try:
        agent = build_agent(dataset_manager)
        logger.info("Agent initialized successfully")
        print("\n Agent initialized successfully!\n")
    except Exception as error:
        logger.error("Agent initialization failed: %s", error)
        print(f"\n Agent initialization failed: {error}")
        return

    # Step 5: Interactive conversation loop
    print("You can now ask questions about your dataset.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n Goodbye!")
            return

        if user_query.lower() in ("exit", "quit"):
            print("\n Goodbye!")
            return

        if not user_query:
            continue

        logger.info("User query: %s", user_query[:100])

        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_query}]}
            )

            final_message = response["messages"][-1]
            content = final_message.content if hasattr(final_message, "content") else str(final_message)

            print("\n========== AGENT RESPONSE ==========\n")
            print(content)
            print("\n====================================\n")

        except KeyboardInterrupt:
            print("\n\n Interrupted. Goodbye!")
            return
        except Exception as error:
            logger.error("Agent execution failed: %s", error)
            print(f"\n Agent execution failed: {error}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Goodbye!")
        sys.exit(0)