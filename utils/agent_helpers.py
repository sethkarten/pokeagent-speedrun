"""
Shared helper functions for all agents.
"""

import logging
import requests
from typing import Dict, Any
from utils.llm_logger import get_llm_logger

logger = logging.getLogger(__name__)


def update_server_metrics(server_url: str = "http://localhost:8000") -> None:
    """
    Update server with current agent step count and LLM metrics.

    This is a shared function used by all agents to send metrics to the server
    for display in the web interface.

    Args:
        server_url: Base URL of the server (default: http://localhost:8000)
    """
    try:

        # Get current LLM metrics
        llm_logger = get_llm_logger()
        metrics = llm_logger.get_cumulative_metrics()

        # Send metrics to server
        try:
            response = requests.post(
                f"{server_url}/agent_step",
                json={"metrics": metrics},
                timeout=1
            )
            if response.status_code != 200:
                logger.debug(f"Failed to update server metrics: {response.status_code}")
        except requests.exceptions.RequestException:
            # Silent fail - server might not be running or in different mode
            pass

    except Exception as e:
        logger.debug(f"Error updating server metrics: {e}")
