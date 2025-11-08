#!/usr/bin/env python3
"""
Example usage of the DeepL Lyrics Translator library
"""

import os
from translation_library import DeepLTranslator, LyricFormatter
from dotenv import load_dotenv

load_dotenv()

def example_basic_translation():
    """Example 1: Basic translation"""
    print("=" * 50)
    print("Example 1: Basic Translation")
    print("=" * 50)
    
    # Initialize translator (make sure DEEPL_API_KEY is set)
    translator = DeepLTranslator(use_cache=False)  # Disable cache for demo
    
    lyrics = """
    I'm walking on sunshine
    And don't it feel good
    Hey, alright now
    And don't it feel good
    """
    
    result = translator.translate_lyrics(
        text=lyrics,
        target_lang='ES'  # Translate to Spanish
    )
    
    print(f"Original ({result['detected_language']}):")
    print(result['original_text'])
    print(f"\nTranslated ({result['target_language']}):")
    print(result['translated_text'])
    print()


def example_multiple_languages():
    """Example 2: Translate to multiple languages"""
    print("=" * 50)
    print("Example 2: Multiple Languages")
    print("=" * 50)
    
    translator = DeepLTranslator(use_cache=False)
    
    lyrics = "Hello, how are you doing today?"
    target_languages = ['ES', 'FR', 'DE', 'IT', 'JA']
    
    print(f"Original: {lyrics}\n")
    
    for lang in target_languages:
        result = translator.translate_lyrics(lyrics, target_lang=lang)
        lang_name = translator.SUPPORTED_LANGUAGES.get(lang, lang)
        print(f"{lang_name} ({lang}): {result['translated_text']}")
    print()


def example_long_lyrics():
    """Example 3: Handle long lyrics with formatter"""
    print("=" * 50)
    print("Example 3: Long Lyrics with Formatter")
    print("=" * 50)
    
    translator = DeepLTranslator(use_cache=False)
    formatter = LyricFormatter()
    
    # Simulate long lyrics
    long_lyrics = """
    Verse 1:
    This is the first verse of our song
    It tells a story that goes on and on
    About dreams and hopes and fears
    That we've been holding through the years
    
    Chorus:
    So sing along with me tonight
    Let's make everything alright
    Together we can reach the stars
    No matter who we are
    
    Verse 2:
    In the morning when the sun rises high
    We'll spread our wings and learn to fly
    Through the clouds and past the moon
    We'll be dancing to our tune
    
    Chorus:
    So sing along with me tonight
    Let's make everything alright
    Together we can reach the stars
    No matter who we are
    """
    
    # Preprocess
    clean_lyrics = formatter.preprocess_lyrics(long_lyrics)
    
    # Split into segments
    segments = formatter.split_into_segments(clean_lyrics, max_segment_length=200)
    print(f"Split into {len(segments)} segments\n")
    
    # Translate each segment
    translations = translator.translate_batch_lyrics(segments, target_lang='FR')
    
    # Reassemble
    final_translation = formatter.reassemble_segments(translations)
    
    print("Original:")
    print(clean_lyrics[:150] + "...")
    print("\nTranslated to French:")
    print(final_translation[:150] + "...")
    print()


def example_supported_languages():
    """Example 4: List supported languages"""
    print("=" * 50)
    print("Example 4: Supported Languages")
    print("=" * 50)
    
    translator = DeepLTranslator(use_cache=False)
    
    languages = translator.get_supported_languages()
    print(f"Total supported languages: {len(languages)}\n")
    
    # Group by first letter for better display
    print("Sample languages:")
    for lang in languages[:10]:
        print(f"  {lang['code']:6} - {lang['name']}")
    print(f"  ... and {len(languages) - 10} more")
    print()


def example_error_handling():
    """Example 5: Error handling"""
    print("=" * 50)
    print("Example 5: Error Handling")
    print("=" * 50)
    
    from translation_library import InvalidLanguageError, TranslationError
    
    translator = DeepLTranslator(use_cache=False)
    
    # Try invalid language
    try:
        translator.translate_lyrics("Hello", target_lang="INVALID")
    except InvalidLanguageError as e:
        print(f"✓ Caught InvalidLanguageError: {e}")
    
    # Try without API key
    try:
        translator_no_key = DeepLTranslator(api_key="")
    except TranslationError as e:
        print(f"✓ Caught TranslationError: {e}")
    
    print()


def example_with_source_language():
    """Example 6: Specify source language"""
    print("=" * 50)
    print("Example 6: Specify Source Language")
    print("=" * 50)
    
    translator = DeepLTranslator(use_cache=False)
    
    french_text = "Bonjour, comment allez-vous?"
    
    result = translator.translate_lyrics(
        text=french_text,
        source_lang='FR',
        target_lang='EN',
        formality='prefer_more'  # More formal translation
    )
    
    print(f"Source (FR): {result['original_text']}")
    print(f"Translation (EN): {result['translated_text']}")
    print()


def main():
    """Run all examples"""
    # Check if API key is set (loaded from .env by load_dotenv)
    if not os.environ.get('DEEPL_API_KEY'):
        print("⚠️  WARNING: DEEPL_API_KEY environment variable not set!")
        print("Please set it with: export DEEPL_API_KEY='your-api-key' or add DEEPL_API_KEY=your-api-key to a .env file")
        print("\nRunning examples that don't require API calls...\n")
        example_supported_languages()
        example_error_handling()
        return
    
    print("\n🎵 DeepL Lyrics Translator - Examples 🎵\n")
    
    try:
        example_basic_translation()
        example_multiple_languages()
        example_long_lyrics()
        example_supported_languages()
        example_with_source_language()
        example_error_handling()
        
        print("=" * 50)
        print("✓ All examples completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure your DEEPL_API_KEY is valid and you have internet connection.")


if __name__ == "__main__":
    main()