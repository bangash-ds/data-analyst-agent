import logging
import sys

from .service import AnalystService

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the AI Data Analyst Agent CLI."""
    print("\n====================================")
    print("       AI DATA ANALYST AGENT")
    print("====================================\n")

    service = AnalystService()

    # Step 1: Get dataset path from user (interface-only validation)
    while True:
        raw_input = input("Enter your dataset path (or 'exit' to quit): ").strip()
        dataset_path = raw_input.strip('\"\'()[]{}').strip()

        if dataset_path.lower() in ("exit", "quit"):
            print("\nGoodbye!")
            return

        if not dataset_path:
            print("Error: Path cannot be empty. Please try again.")
            continue

        # Delegate actual file validation to service layer
        load_result = service.load_dataset(dataset_path)
        if load_result["status"] == "loaded":
            print(f"\n Dataset loaded successfully: {load_result['rows']} rows\n")
            break
        else:
            print(f"\n {load_result['error']}\n")

    # Step 2: Initialize agent via service
    init_result = service.initialize_agent()
    if init_result["status"] != "ready":
        print(f"\n {init_result['error']}")
        return

    print("You can now ask questions about your dataset.")
    print("Type 'exit' or 'quit' to stop.\n")

    # Step 3: Interactive conversation loop (interface-only)
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

        try:
            answer_result = service.ask(user_query)
            print("\n========== AGENT RESPONSE ==========\n")
            print(answer_result["answer"])
            print("\n====================================\n")

        except KeyboardInterrupt:
            print("\n\n Interrupted. Goodbye!")
            return
        except Exception as error:
            print(f"\n Agent execution failed: {error}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Goodbye!")
        sys.exit(0)