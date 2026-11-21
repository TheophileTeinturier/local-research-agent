"""Local research agent: a local LLM (Ollama) that calls Python tools via native tool calling."""

import logging

# Library convention: emit nothing unless the application (the CLI) configures logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "1.0.0rc1"  # keep in sync with pyproject.toml
