# === DESCRIPTION ======================================================================================================
"""
Provides path utilities for all other files.
"""


# === DEPENDENCIES =====================================================================================================
from pathlib import Path


# === CONSTANTS ========================================================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent


# === FUNCTIONS ========================================================================================================
def path(
        abs_path: str  # Path from project root
):
    """
    Resolves a path from project root so that it is relative to the current file.
    :param abs_path: Absolute path from project root
    :return: Path
    """

    # Return resolved path
    return PROJECT_ROOT/abs_path

