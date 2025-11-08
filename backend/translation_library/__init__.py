"""
DeepL Lyrics Translator Library

A Python library for translating song lyrics using the DeepL API,
with caching, formatting, and music-specific optimizations.
"""

from .deepl_translator import DeepLTranslator
from .lyric_formatter import LyricFormatter
from .translation_cache import TranslationCache
from .exceptions import TranslationError, RateLimitError, InvalidLanguageError

__version__ = "0.1.0"
__all__ = [
    "DeepLTranslator",
    "LyricFormatter",
    "TranslationCache",
    "TranslationError",
    "RateLimitError",
    "InvalidLanguageError",
]