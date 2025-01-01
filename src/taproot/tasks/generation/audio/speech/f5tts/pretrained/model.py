from __future__ import annotations

from typing import Any, Dict, Optional, Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from taproot.modeling import ConditionalFlowMatching

__all__ = ["F5TTSModel"]

class F5TTSModel(PretrainedModelMixin):
    """
    F5TTS model, a conditional flow matching (CFM) model with a DiT backbone.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/speech-synthesis-f5tts.safetensors"
    init_file_urls = {
        "vocab": "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/speech-synthesis-f5tts-vocab.txt",
    }

    @classmethod
    def get_model_class(cls) -> Type[ConditionalFlowMatching]:
        """
        Returns the model class.
        """
        from taproot.modeling import ConditionalFlowMatching
        return ConditionalFlowMatching

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        return {
            "dim": 1024,
            "depth": 22,
            "heads": 16,
            "ff_mult": 2,
            "text_dim": 512,
            "conv_layers": 4
        }

    @classmethod
    def instantiate_model(cls, **kwargs: Any) -> Any:
        """
        Instantiates the model.
        """
        model_class = cls.get_model_class()
        config = cls.merge_default_config(kwargs)

        assert isinstance(config, dict), "Config must be a dictionary."
        config_file = config.pop("vocab", None)
        assert isinstance(config_file, str), "Must pass vocab file path as 'vocab'."

        vocab: Dict[str, int] = {}
        with open(config_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                vocab[line[:-1]] = i

        config["vocab_char_map"] = vocab
        return model_class.dit(**config)
