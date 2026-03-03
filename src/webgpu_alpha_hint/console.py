"""Shared Rich console and logging for internal use.

Logging setup is deferred to avoid hijacking the consumer's logging
configuration on import. Call ``setup_logging()`` explicitly (the CLI
does this) or just use ``log`` — it works with whatever handler the
consumer has already configured.
"""

import logging

from rich.console import Console

console = Console()

log = logging.getLogger("webgpu-alpha-hint")


def setup_logging() -> None:
    """Configure Rich-backed logging. Only call from CLI entry points."""
    from rich.logging import RichHandler

    if not log.handlers:
        handler = RichHandler(console=console, rich_tracebacks=True, show_path=False)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
