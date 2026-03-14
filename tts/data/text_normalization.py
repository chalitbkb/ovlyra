"""Input text normalization."""

import abc
import logging as base_logging
import re

import lingua
import unidecode
from nemo_text_processing.text_normalization import normalize
from pythainlp.util import normalize as thai_normalize
from pythainlp.util import num_to_thaiword

base_logging.getLogger("NeMo-text-processing").setLevel(base_logging.CRITICAL)

# Language codes for text normalization.
_ENGLISH = "en"
_JAPANESE = "ja"
_CHINESE = "zh"
_SPANISH = "es"
_FRENCH = "fr"
_GERMAN = "de"
_THAI = "th"


class TextNormalizer(metaclass=abc.ABCMeta):
    """Text normalization class for normalizers to implement."""

    @abc.abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize"""
        raise NotImplementedError("|normalize| is not implemented.")

    @abc.abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Get supported languages."""
        raise NotImplementedError("|get_supported_languages| is not implemented.")

    @abc.abstractmethod
    def normalize_with_language(self, text: str, language: str) -> str:
        """Normalize text with a specific language."""
        raise NotImplementedError("|normalize_with_language| is not implemented.")


class NoOpTextNormalizer(TextNormalizer):
    """No-op text normalizer."""

    def normalize(self, text: str) -> str:
        return text

    def get_supported_languages(self) -> list[str]:
        return []

    def normalize_with_language(self, text: str, language: str) -> str:
        return text


class NemoTextNormalizer(TextNormalizer):
    """Text normalizer for different languages using Nvidias NeMo text normalization
    library."""

    def __init__(self):
        super().__init__()
        # Languages supported by NeMo text normalization.
        self._nemo_languages = [
            _ENGLISH,
            _JAPANESE,
            _CHINESE,
            _SPANISH,
            _FRENCH,
            _GERMAN,
        ]
        self._supported_languages = self._nemo_languages + [_THAI]
        self._normalize_text = {
            lang: normalize.Normalizer(input_case="cased", lang=lang)
            for lang in self._nemo_languages
        }
        self.lang_detector = None

    def init_lang_detector(self):
        self.lang_detector = lingua.LanguageDetectorBuilder.from_languages(
            lingua.Language.KOREAN,
            lingua.Language.JAPANESE,
            lingua.Language.CHINESE,
            lingua.Language.ENGLISH,
            lingua.Language.SPANISH,
            lingua.Language.FRENCH,
            lingua.Language.GERMAN,
            lingua.Language.THAI,
        ).build()

    def convert_to_ascii(self, text: str) -> str:
        return unidecode.unidecode(text)

    def get_supported_languages(self) -> list[str]:
        return self._supported_languages

    def normalize(self, text: str) -> str:
        # detect language and normalize text
        try:
            # Only initialize the language detector if it's not already initialized
            # (dynamic language detection).
            if self.lang_detector is None:
                self.init_lang_detector()
            language = self.lang_detector.detect_language_of(text)
            if language == lingua.Language.ENGLISH:
                return self.normalize_with_language(text, _ENGLISH)
            elif language == lingua.Language.JAPANESE:
                return self.normalize_with_language(text, _JAPANESE)
            elif language == lingua.Language.CHINESE:
                return self.normalize_with_language(text, _CHINESE)
            elif language == lingua.Language.SPANISH:
                return self.normalize_with_language(text, _SPANISH)
            elif language == lingua.Language.FRENCH:
                return self.normalize_with_language(text, _FRENCH)
            elif language == lingua.Language.GERMAN:
                return self.normalize_with_language(text, _GERMAN)
            elif language == lingua.Language.THAI:
                return self.normalize_with_language(text, _THAI)
            else:
                return text
        except Exception:
            return text

    def _normalize_thai(self, text: str) -> str:
        """Normalize Thai text using pythainlp."""
        # Normalize Thai characters (fix combining characters, sara issues, etc.)
        text = thai_normalize(text)
        # Convert Arabic numerals to Thai words (e.g., "123" → "หนึ่งร้อยยี่สิบสาม")
        text = re.sub(r"\d+", lambda m: num_to_thaiword(int(m.group())), text)
        return text

    def normalize_with_language(self, text: str, language: str) -> str:
        if language not in self._supported_languages:
            return text

        if language == _ENGLISH:
            text = self.convert_to_ascii(text)

        if language == _THAI:
            try:
                return self._normalize_thai(text)
            except Exception:
                return text

        try:
            text = self._normalize_text[language].normalize(text)
        except Exception:
            # return the unnormalized text if error
            return text
        return text


def create_text_normalizer(enable_text_normalization: bool) -> TextNormalizer:
    """Create text normalizer for NemoNormalizer."""
    if enable_text_normalization:
        return NemoTextNormalizer()
    else:
        return NoOpTextNormalizer()
