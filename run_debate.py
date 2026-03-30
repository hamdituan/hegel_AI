"""Debate orchestrator script - runs multi-agent philosophical debates."""

import sys

from hegel_ai.config import get_config
from hegel_ai.logging_config import setup_logging
from hegel_ai.debate.orchestrator import run_debate


if __name__ == "__main__":
    setup_logging()

    try:
        run_debate()

    except KeyboardInterrupt:
        import logging
        logging.info("\nDebate interrupted by user.")
    except Exception as e:
        import logging
        logging.exception(f"Debate failed: {e}")
        sys.exit(1)
