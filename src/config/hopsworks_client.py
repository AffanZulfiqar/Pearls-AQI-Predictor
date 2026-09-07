"""
Centralized Hopsworks connection module.

ALL components that need Hopsworks must import from here.
No inline hopsworks.login() calls anywhere else in the project.
"""
import logging
import hopsworks
from src.config import settings

logger = logging.getLogger(__name__)

_project = None  # Cached project connection


def get_project():
    """
    Returns a connected Hopsworks project, caching the connection.
    On failure, raises the exception — never silently returns None.
    """
    global _project
    if _project is not None:
        return _project

    logger.info("Initializing Hopsworks connection (engine=python)...")
    try:
        _project = hopsworks.login(
            project=settings.HOPSWORKS_PROJECT_NAME,
            api_key_value=settings.HOPSWORKS_API_KEY
        )
        logger.info("Hopsworks connection established: project=%s", settings.HOPSWORKS_PROJECT_NAME)
        return _project
    except Exception:
        logger.exception("Hopsworks login failed")
        raise


def get_feature_store():
    """Returns the Hopsworks Feature Store for the configured project."""
    project = get_project()
    return project.get_feature_store()


def get_model_registry():
    """Returns the Hopsworks Model Registry for the configured project."""
    project = get_project()
    return project.get_model_registry()


def reset_connection():
    """
    Resets the cached connection. Useful in tests or when the connection
    needs to be re-established after a timeout.
    """
    global _project
    _project = None
