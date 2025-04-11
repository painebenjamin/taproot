from __future__ import annotations

import re
import torch
import torch.nn as nn

from typing import Dict, List, TYPE_CHECKING

from taproot.util import normalize_text

if TYPE_CHECKING:
    import phonemizer # type: ignore[import-untyped]

__all__ = [
    "init_weights",
    "get_padding",
    "length_to_mask",
    "tokenize",
    "untokenize",
    "phonemize",
]

def init_weights(
    m: nn.Module,
    mean: float=0.0,
    std: float=0.01
) -> None:
    """
    Initialize the weights of a module with a normal distribution.

    :param m: The module to initialize.
    :param mean: The mean of the normal distribution.
    :param std: The standard deviation of the normal distribution.
    """
    if "conv" in m.__class__.__name__.lower():
        m.weight.data.normal_(mean, std) # type: ignore[operator]

def get_padding(
    kernel_size: int,
    dilation: int=1
) -> int:
    """
    Calculate the padding required to maintain the input size.

    :param kernel_size: The size of the kernel.
    :param dilation: The dilation factor of the kernel.
    :return: The padding required to maintain the input size
    """
    return int((kernel_size * dilation - dilation) / 2)

def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor of lengths to a mask.

    :param lengths: The lengths of the sequences. [batch_size]
    :return: The mask. [batch_size, max_length]
    """
    mask = torch.arange(lengths.max().item()) \
                .unsqueeze(0) \
                .expand(lengths.shape[0], -1) \
                .type_as(lengths)

    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def get_vocab() -> Dict[str, int]:
    """
    Gets the vocabulary for the dataset.
    """
    pad = "$"
    punctuation = ";:,.!?¡¿—…\"«»“” "
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    symbols = [pad] + list(punctuation) + list(letters) + list(letters_ipa)
    dicts = {}
    for i in range(len((symbols))):
        dicts[symbols[i]] = i
    return dicts

VOCAB = get_vocab()
def tokenize(text: str) -> List[int]:
    """
    Tokenizes the text into a list of integers.

    :param text: The text to tokenize.
    :return: The list of integers.
    """
    return [i for i in map(VOCAB.get, text) if i is not None]

def untokenize(tokens: List[int]) -> str:
    """
    Untokenizes the list of integers into a string.

    :param tokens: The list of integers.
    :return: The string.
    """
    return "".join(next(k for k, v in VOCAB.items() if v == i) for i in tokens)

PHONEMIZERS: Dict[str, phonemizer.backend.EspeakBackend] = {}
def get_phonemizer(lang: str) -> phonemizer.backend.EspeakBackend:
    """
    Get the phonemizer for the specified language.

    :param lang: The language to use.
    :return: The phonemizer.
    """
    if lang not in PHONEMIZERS:
        import phonemizer
        PHONEMIZERS[lang] = phonemizer.backend.EspeakBackend(
            language=lang,
            preserve_punctuation=True,
            with_stress=False,
        )
    return PHONEMIZERS[lang]

def phonemize(
    text: str,
    lang: str,
    normalize: bool=False
) -> str:
    """
    Phonemize the text using the specified language.

    :param text: The text to phonemize.
    :param lang: The language to use.
    :return: The phonemized text.
    """
    if normalize:
        text = normalize_text(text)
    phonemizer = get_phonemizer(lang)
    phonemized = phonemizer.phonemize([text])
    phonemes = phonemized[0] if phonemized else ""
    # Fix pronunciation errors for the name of the model (kokoro)
    phonemes = phonemes.replace("kəkˈoːɹoʊ", "kˈoʊkəɹoʊ") \
                       .replace("kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ") \
                       .replace("ʲ", "j") \
                       .replace("r", "ɹ") \
                       .replace("x", "k") \
                       .replace("ɬ", "l")
    phonemes = re.sub(r"(?<=[a-zɹː])(?=hˈʌndɹɪd)", " ", phonemes)
    phonemes = re.sub(r" z(?=[;:,.!?¡¿—…\"«»“” ]|$)", "z", phonemes)
    if lang == "en-us":
        phonemes = re.sub(r"(?<=nˈaɪn)ti(?!ː)", "di", phonemes)
    phonemes = "".join(filter(lambda p: p in VOCAB, phonemes))
    return phonemes.strip()
