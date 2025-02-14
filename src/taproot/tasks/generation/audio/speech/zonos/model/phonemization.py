# Adapted from https://github.com/Zyphra/Zonos
from __future__ import annotations

from typing import List, Tuple, Dict, TYPE_CHECKING

from taproot.util import normalize_text, normalize_jp_text

if TYPE_CHECKING:
    import torch
    from phonemizer.backend import EspeakBackend # type: ignore[import-untyped]

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

SPECIAL_TOKEN_IDS = [PAD_ID, UNK_ID, BOS_ID, EOS_ID]
PUNCTUATION = ';:,.!?¡¿—…"«»“”() *~-/\\&'

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
ALPHABET_IPA = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

SYMBOLS = [*PUNCTUATION, *ALPHABET, *ALPHABET_IPA]
SYMBOL_TO_ID = {
    s: i
    for i, s in enumerate(
        SYMBOLS,
        start=len(SPECIAL_TOKEN_IDS)
    )
}
NUM_SYMBOLS = len(SPECIAL_TOKEN_IDS) + len(SYMBOLS)

LANGUAGES = [
    'af', 'am', 'an', 'ar', 'as', 'az', 'ba', 'bg', 'bn', 'bpy', 'bs', 'ca', 'cmn',
    'cs', 'cy', 'da', 'de', 'el', 'en-029', 'en-gb', 'en-gb-scotland', 'en-gb-x-gbclan',
    'en-gb-x-gbcwmd', 'en-gb-x-rp', 'en-us', 'eo', 'es', 'es-419', 'et', 'eu', 'fa',
    'fa-latn', 'fi', 'fr-be', 'fr-ch', 'fr-fr', 'ga', 'gd', 'gn', 'grc', 'gu', 'hak',
    'hi', 'hr', 'ht', 'hu', 'hy', 'hyw', 'ia', 'id', 'is', 'it', 'ja', 'jbo', 'ka',
    'kk', 'kl', 'kn', 'ko', 'kok', 'ku', 'ky', 'la', 'lfn', 'lt', 'lv', 'mi', 'mk',
    'ml', 'mr', 'ms', 'mt', 'my', 'nb', 'nci', 'ne', 'nl', 'om', 'or', 'pa', 'pap',
    'pl', 'pt', 'pt-br', 'py', 'quc', 'ro', 'ru', 'ru-lv', 'sd', 'shn', 'si', 'sk',
    'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'tn', 'tr', 'tt', 'ur', 'uz', 'vi',
    'vi-vn-x-central', 'vi-vn-x-south', 'yue'
]
LANGUAGE_TO_ID = {
    l: i
    for i, l in enumerate(LANGUAGES)
}
NUM_LANGUAGES = len(LANGUAGES)

__all__ = [
    "get_language_id",
    "get_symbol_id",
    "get_symbol_ids",
    "get_espeak_backend",
    "phonemize",
    "tokenize"
]

def get_language_id(language: str) -> int:
    """
    :param language: A string representing a language
    :return: The id of the language
    :raises KeyError: If the language is not supported
    """
    return LANGUAGE_TO_ID[language]

def get_symbol_id(symbol: str) -> int:
    """
    :param symbol: A string representing a symbol
    :return: The id of the symbol
    """
    return SYMBOL_TO_ID.get(symbol, UNK_ID)

def get_symbol_ids(symbols: List[str]) -> List[int]:
    """
    :param symbols: A string representing a sequence of symbols
    :return: The ids of the symbols
    """
    return [get_symbol_id(s) for s in symbols]

ESPEAK_BACKENDS: Dict[str, EspeakBackend] = {}
def get_espeak_backend(
    language: str,
    preserve_punctuation: bool = True,
    with_stress: bool = True,
    punctuation_marks: str = PUNCTUATION
) -> EspeakBackend:
    """
    :param language: The language of the backend
    :return: The backend for the language
    """
    if language not in ESPEAK_BACKENDS:
        from phonemizer.backend import EspeakBackend
        ESPEAK_BACKENDS[language] = EspeakBackend(
            language=language,
            preserve_punctuation=preserve_punctuation,
            with_stress=with_stress,
            punctuation_marks=punctuation_marks
        )
    return ESPEAK_BACKENDS[language]

def phonemize(text: str, language: str, normalize: bool=False) -> List[str]:
    """
    :param text: The text to phonemize
    :param language: The language of the text
    :return: The phonemized text
    """
    if normalize:
        if "ja" in language:
            text = normalize_jp_text(text)
        else:
            text = normalize_text(text)

    backend = get_espeak_backend(language)
    [phonemes] = backend.phonemize([text], strip=True)

    return phonemes # type: ignore[no-any-return]

def tokenize(phonemes: List[List[str]]) -> Tuple[torch.Tensor, List[int]]:
    """
    :param phonemes: A list of phonemes
    :return: The tokenized phonemes and the lengths of the phonemes
    """
    import torch
    phoneme_ids = [[BOS_ID, *get_symbol_ids(p), EOS_ID] for p in phonemes]
    lengths = list(map(len, phoneme_ids))
    longest = max(lengths)
    phoneme_ids = [[PAD_ID] * (longest - len(p)) + p for p in phoneme_ids]
    return torch.tensor(phoneme_ids), lengths
