# Adapted from https://github.com/Zyphra/Zonos
import torch
import torch.nn as nn

from abc import ABCMeta, abstractmethod
from typing import Any, Iterable, Optional, Tuple, Union, List, Dict
from typing_extensions import Literal

from .config import PrefixConditionerConfig
from .phonemization import phonemize, tokenize, get_language_id, NUM_SYMBOLS

__all__ = [
    "Conditioner",
    "EspeakPhonemeConditioner",
    "FourierConditioner",
    "IntegerConditioner",
    "PassthroughConditioner",
    "PrefixConditioner",
    "make_cond_dict",
]

class Conditioner(nn.Module, metaclass=ABCMeta):
    project: nn.Module

    def __init__(
        self,
        output_dim: int,
        name: str,
        cond_dim: Optional[int]=None,
        projection: Optional[Literal["none", "linear", "mlp"]]=None,
        uncond_type: Optional[Literal["learned", "none"]]=None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.name = name
        self.output_dim = output_dim
        self.cond_dim = cond_dim = cond_dim or output_dim

        if projection == "linear":
            self.project = nn.Linear(cond_dim, output_dim)
        elif projection == "mlp":
            self.project = nn.Sequential(
                nn.Linear(cond_dim, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
            )
        else:
            self.project = nn.Identity()

        self.uncond_vector = None
        if uncond_type == "learned":
            self.uncond_vector = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self,
        inputs: Optional[Tuple[Any, ...]]
    ) -> torch.Tensor:
        if inputs is None:
            assert self.uncond_vector is not None
            return self.uncond_vector.data.view(1, 1, -1)

        cond = self.apply_cond(*inputs)
        cond = self.project(cond)
        return cond

    @abstractmethod
    def apply_cond(self, *inputs: Any) -> torch.Tensor:
        pass

class EspeakPhonemeConditioner(Conditioner):
    """
    A conditioner that converts text to phonemes using eSpeak.
    """
    def __init__(
        self,
        output_dim: int,
        **kwargs: Any
    ) -> None:
        super().__init__(output_dim, **kwargs)
        self.phoneme_embedder = nn.Embedding(NUM_SYMBOLS, output_dim)

    def apply_cond(
        self,
        texts: List[str],
        languages: List[str]
    ) -> torch.Tensor:
        """
        :param texts: list of texts to convert to phonemes
        :param languages: ISO 639-1 -or otherwise eSpeak compatible-language code
        """
        device = self.phoneme_embedder.weight.device

        phonemes: List[List[str]] = []
        for text, lang in zip(texts, languages):
            phonemes.append(phonemize(text, lang))

        phoneme_ids, lengths = tokenize(phonemes)
        phoneme_embeds = self.phoneme_embedder(phoneme_ids.to(device))

        return phoneme_embeds # type: ignore[no-any-return]

class FourierConditioner(Conditioner):
    def __init__(
        self,
        output_dim: int,
        input_dim: int = 1,
        std: float = 1.0,
        min_val: float = 0.0,
        max_val: float = 1.0,
        **kwargs: Any
    ) -> None:
        assert output_dim % 2 == 0
        super().__init__(output_dim, **kwargs)
        self.register_buffer("weight", torch.randn([output_dim // 2, input_dim]) * std)
        self.input_dim, self.min_val, self.max_val = input_dim, min_val, max_val

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.input_dim
        x = (x - self.min_val) / (self.max_val - self.min_val)  # [batch_size, seq_len, input_dim]
        f = 2 * torch.pi * x.to(self.weight.dtype) @ self.weight.T  # type: ignore[operator,arg-type]
        return torch.cat([f.cos(), f.sin()], dim=-1)  # [batch_size, seq_len, output_dim]

class IntegerConditioner(Conditioner):
    def __init__(
        self,
        output_dim: int,
        min_val: int = 0,
        max_val: int = 512,
        **kwargs: Any
    ) -> None:
        super().__init__(output_dim, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim)

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 1
        return self.int_embedder(x.squeeze(-1) - self.min_val)  # type: ignore[no-any-return]

class PassthroughConditioner(Conditioner):
    def __init__(self, output_dim: int, **kwargs: Any) -> None:
        super().__init__(output_dim, **kwargs)

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.cond_dim
        return x

def build_conditioner(config: Dict[str, Any], output_dim: int) -> Conditioner:
    """
    Builds a conditioner from a configuration dictionary.
    """
    if config["type"] == "PassthroughConditioner":
        return PassthroughConditioner(output_dim, **config)
    elif config["type"] == "EspeakPhonemeConditioner":
        return EspeakPhonemeConditioner(output_dim, **config)
    elif config["type"] == "FourierConditioner":
        return FourierConditioner(output_dim, **config)
    elif config["type"] == "IntegerConditioner":
        return IntegerConditioner(output_dim, **config)
    raise ValueError(f"Unknown conditioner type: {config['type']}")

def build_conditioners(
    conditioners: List[Dict[str, Any]],
    output_dim: int
) -> List[Conditioner]:
    """
    Builds a list of conditioners from a list of configuration dictionaries.
    """
    return [
        build_conditioner(c, output_dim)
        for c in conditioners
    ]

class PrefixConditioner(Conditioner):
    """
    A meta-conditioner that concatenates the outputs of multiple conditioners.
    """
    def __init__(
        self,
        config: PrefixConditionerConfig,
        output_dim: int
    ) -> None:
        super().__init__(output_dim, "prefix", projection=config.projection)
        self.conditioners = nn.ModuleList(build_conditioners(config.conditioners, output_dim))
        self.norm = nn.LayerNorm(output_dim)
        self.required_keys = {c.name for c in self.conditioners if c.uncond_vector is None}

    def apply_cond(self, cond_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Applies the conditioners to the input dictionary and concatenates the results.
        """
        if not set(cond_dict).issuperset(self.required_keys):
            raise ValueError(f"Missing required keys: {self.required_keys - set(cond_dict)}")
        conds = []
        for conditioner in self.conditioners:
            conds.append(conditioner(cond_dict.get(conditioner.name)))
        max_bsz = max(map(len, conds))
        assert all(c.shape[0] in (max_bsz, 1) for c in conds)
        conds = [c.expand(max_bsz, -1, -1) for c in conds]
        return self.norm(self.project(torch.cat(conds, dim=-2))) # type: ignore[no-any-return]

    def forward(self, cond_dict: Dict[str, Any]) -> torch.Tensor: # type: ignore[override]
        return self.apply_cond(cond_dict)

def make_cond_dict(
    text: str,
    language: str,
    speaker: Optional[torch.Tensor] = None,
    emotion: List[float] = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077],
    fmax: float = 22050.0,
    pitch_std: float = 20.0,
    speaking_rate: float = 15.0,
    vqscore_8: List[float] = [0.78] * 8,
    ctc_loss: float = 0.0,
    dnsmos_ovrl: float = 4.0,
    speaker_noised: bool = False,
    unconditional_keys: Iterable[str] = {"vqscore_8", "dnsmos_ovrl"},
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    """
    A helper to build the 'cond_dict' that the model expects.
    """
    cond_dict = {
        "espeak": ([text], [language]),
        "speaker": speaker,
        "emotion": emotion,
        "fmax": fmax,
        "pitch_std": pitch_std,
        "speaking_rate": speaking_rate,
        "language_id": get_language_id(language),
        "vqscore_8": vqscore_8,
        "ctc_loss": ctc_loss,
        "dnsmos_ovrl": dnsmos_ovrl,
        "speaker_noised": int(speaker_noised),
    }

    if dtype is None:
        dtype = torch.float32

    for k in unconditional_keys:
        cond_dict.pop(k, None)

    for k, v in cond_dict.items():
        if isinstance(v, (float, int, list)):
            v = torch.tensor(v)

        if isinstance(v, torch.Tensor) and device is not None:
            cond_dict[k] = v.view(1, 1, -1).to(device)
            if v.is_floating_point():
                cond_dict[k] = cond_dict[k].to(dtype) # type: ignore[union-attr]

        if k == "emotion":
            if not isinstance(v, torch.Tensor):
                cond_dict[k] = torch.tensor(v, dtype=dtype, device=device).view(1, 1, -1)

            cond_dict[k] /= cond_dict[k].sum(dim=-1) # type: ignore[union-attr]

    return cond_dict
