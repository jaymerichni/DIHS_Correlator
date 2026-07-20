"""Flask application entrypoints for the DIHS Tephra Correlator."""

from DIHS_Correlator.web.app import app, create_app, main

__all__ = ["app", "create_app", "main"]
