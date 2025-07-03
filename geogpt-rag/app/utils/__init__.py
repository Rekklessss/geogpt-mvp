"""
Utilities package for GeoGPT-RAG.

This package provides text processing and parsing utilities including
intelligent document splitting using BERT-based sentence prediction.
"""

from .parsers import TextSplitter, split_text

__all__ = [
    "TextSplitter",
    "split_text",
]
