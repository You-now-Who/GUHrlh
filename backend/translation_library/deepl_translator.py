import os
import requests
from typing import Dict, List, Optional
from .translation_cache import TranslationCache
from .exceptions import TranslationError, RateLimitError, InvalidLanguageError

class DeepLTranslator:
    # Supported DeepL languages (as of 2024)
    SUPPORTED_LANGUAGES = {
        'AR': 'Arabic', 'BG': 'Bulgarian', 'CS': 'Czech', 'DA': 'Danish',
        'DE': 'German', 'EL': 'Greek', 'EN': 'English', 'EN-GB': 'English (British)',
        'EN-US': 'English (American)', 'ES': 'Spanish', 'ET': 'Estonian',
        'FI': 'Finnish', 'FR': 'French', 'HU': 'Hungarian', 'ID': 'Indonesian',
        'IT': 'Italian', 'JA': 'Japanese', 'KO': 'Korean', 'LT': 'Lithuanian',
        'LV': 'Latvian', 'NB': 'Norwegian', 'NL': 'Dutch', 'PL': 'Polish',
        'PT': 'Portuguese', 'PT-BR': 'Portuguese (Brazilian)', 'PT-PT': 'Portuguese (European)',
        'RO': 'Romanian', 'RU': 'Russian', 'SK': 'Slovak', 'SL': 'Slovenian',
        'SV': 'Swedish', 'TR': 'Turkish', 'UK': 'Ukrainian', 'ZH': 'Chinese (simplified)'
    }
    
    def __init__(self, api_key: Optional[str] = None, use_cache: bool = True):
        self.api_key = api_key or os.getenv('DEEPL_API_KEY')
        if not self.api_key:
            raise TranslationError("DeepL API key is required. Set DEEPL_API_KEY environment variable or pass api_key parameter.")
        
        # Auto-detect API endpoint based on key type
        # Free API keys end with ':fx', Pro keys don't
        if self.api_key.endswith(':fx'):
            self.base_url = "https://api-free.deepl.com/v2/translate"
            self.usage_url = "https://api-free.deepl.com/v2/usage"
        else:
            self.base_url = "https://api.deepl.com/v2/translate"
            self.usage_url = "https://api.deepl.com/v2/usage"
        
        self.use_cache = use_cache
        self.cache = TranslationCache() if use_cache else None
        
    def _validate_language(self, lang_code: str, param_name: str = "language") -> None:
        """Validate language code against supported languages"""
        if lang_code and lang_code.upper() not in self.SUPPORTED_LANGUAGES:
            raise InvalidLanguageError(
                f"Unsupported {param_name}: {lang_code}. "
                f"Supported languages: {', '.join(self.SUPPORTED_LANGUAGES.keys())}"
            )
    
    def translate_lyrics(
        self,
        text: str,
        target_lang: str = 'EN',
        source_lang: Optional[str] = None,
        preserve_formatting: bool = True,
        formality: str = 'prefer_less'  # Better for lyrics
    ) -> Dict:
        """
        Translate lyrics with music-specific optimizations
        
        Args:
            text: The lyrics text to translate
            target_lang: Target language code (e.g., 'EN', 'ES', 'FR')
            source_lang: Source language code (optional, auto-detected if not provided)
            preserve_formatting: Whether to preserve line breaks and formatting
            formality: Formality level ('default', 'prefer_more', 'prefer_less')
        
        Returns:
            Dict containing translation results
        """
        # Validate languages
        self._validate_language(target_lang, "target language")
        if source_lang:
            self._validate_language(source_lang, "source language")
        
        # Check cache first
        cache_key = f"{source_lang or 'auto'}:{target_lang}:{hash(text)}"
        if self.use_cache and self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        # Prepare request
        params = {
            'text': text,
            'target_lang': target_lang.upper(),
            'preserve_formatting': '1' if preserve_formatting else '0',
            'formality': formality
        }
        
        if source_lang:
            params['source_lang'] = source_lang.upper()
        
        headers = {
            'Authorization': f'DeepL-Auth-Key {self.api_key}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        try:
            response = requests.post(
                self.base_url,
                data=params,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 429:
                raise RateLimitError("DeepL API rate limit exceeded")
            elif response.status_code != 200:
                raise TranslationError(f"DeepL API error: {response.status_code}")
            
            data = response.json()
            translation = {
                'original_text': text,
                'translated_text': data['translations'][0]['text'],
                'detected_language': data['translations'][0]['detected_source_language'],
                'target_language': target_lang,
                'confidence': 'high'  # You could add confidence scoring
            }
            
            # Cache the result
            if self.use_cache and self.cache:
                self.cache.set(cache_key, translation)
            return translation
            
        except requests.exceptions.RequestException as e:
            raise TranslationError(f"Network error: {str(e)}")
    
    def translate_batch_lyrics(
        self, 
        texts: List[str], 
        target_lang: str = 'EN'
    ) -> List[Dict]:
        """
        Translate multiple lyric segments efficiently
        """
        return [self.translate_lyrics(text, target_lang) for text in texts]
    
    def get_supported_languages(self) -> List[Dict]:
        """
        Get list of languages supported by DeepL
        """
        return [
            {'code': code, 'name': name}
            for code, name in sorted(self.SUPPORTED_LANGUAGES.items())
        ]
    
    def get_usage_stats(self) -> Dict:
        """
        Get DeepL API usage statistics
        """
        headers = {
            'Authorization': f'DeepL-Auth-Key {self.api_key}'
        }
        
        try:
            response = requests.get(
                self.usage_url,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise TranslationError(f"Failed to get usage stats: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            raise TranslationError(f"Network error: {str(e)}")