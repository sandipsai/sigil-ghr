"""
Sigil Plugins Package.
"""

from sigil.plugins.vlc_plugin import VLCPlugin
from sigil.plugins.vscode_plugin import VSCodePlugin
from sigil.plugins.powerpoint_plugin import PowerPointPlugin
from sigil.plugins.browser_plugin import BrowserPlugin
from sigil.plugins.obs_plugin import OBSPlugin
from sigil.plugins.pdf_plugin import PDFPlugin

__all__ = [
    "VLCPlugin",
    "VSCodePlugin",
    "PowerPointPlugin",
    "BrowserPlugin",
    "OBSPlugin",
    "PDFPlugin",
]
