"""MLSC Knowledge Assistant — a grounded, citation-first RAG system over the MLSC knowledge base."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mlsc-knowledge-assistant")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
