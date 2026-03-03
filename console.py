"""Shared Rich console and logging for consistent CLI output."""

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()

# Rich-backed handler so all stdlib logging renders with markup and timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
)

log = logging.getLogger("webgpu-alpha-hint")
