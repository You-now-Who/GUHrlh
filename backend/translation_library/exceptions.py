class TranslationError(Exception):
    """Base exception for translation errors"""
    pass

class RateLimitError(TranslationError):
    """Raised when API rate limit is exceeded"""
    pass

class InvalidLanguageError(TranslationError):
    """Raised when an unsupported language is requested"""
    pass