from __future__ import annotations

import os

from typing import Any, Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from vocos import Vocos # type: ignore[import-not-found,import-untyped,unused-ignore]

__all__ = ["F5TTSVocoder"]

class F5TTSVocoder(PretrainedModelMixin):
    """
    Vocos mel-scale spectrogram vocoder.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/audio-vocoder-vocos-mel-24khz.safetensors"
    init_file_urls = {
        "config": "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/audio-vocoder-vocos-mel-24khz-config.yaml",
    }

    @classmethod
    def get_model_class(cls) -> Type[Vocos]:
        """
        Returns the model class.
        """
        from vocos import Vocos
        return Vocos # type: ignore[no-any-return]

    @classmethod
    def instantiate_model(cls, **kwargs: Any) -> Any:
        """
        Instantiates the model.

        :see https://github.com/gemelo-ai/vocos/blob/main/vocos/pretrained.py#L50:
        """
        model_class = cls.get_model_class()
        config = cls.merge_default_config(kwargs)

        assert isinstance(config, dict), "Config must be a dictionary."
        assert "config" in config, "Missing config file path."
        assert os.path.exists(config["config"]), "Config file does not exist or cannot be accessed."
        return model_class.from_hparams(config["config"])
