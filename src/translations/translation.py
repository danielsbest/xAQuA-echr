from typing import Optional, Protocol


LANGUAGE_CODES = {
    'eng': 'English',
    'ron': 'Romanian',
    'fra': 'French',
}

class Translator(Protocol):
    def translate(self, text: str, target_lang: str = 'eng', source_lang: Optional[str] = None) -> str:
        """
        Translate text from source_lang to target_lang.

        :param text: The text to translate.
        :param source_lang: The source language code.
        :param target_lang: The target language code.
        :return: The translated text.
        """
        ...