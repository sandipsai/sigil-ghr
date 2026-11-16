"""
Sigil Config Package.

Exposes ConfigServer — a zero-dependency local HTTP server that serves
the HTML+CSS configuration panel and a simple JSON REST API.
"""

from sigil.config.server import ConfigServer

__all__ = ["ConfigServer"]
