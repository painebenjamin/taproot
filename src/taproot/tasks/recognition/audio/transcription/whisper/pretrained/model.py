from __future__ import annotations

from typing import Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration # type: ignore[import-not-found,import-untyped,unused-ignore]
    from transformers.models.whisper.configuration_whisper import WhisperConfig # type: ignore[import-not-found,import-untyped,unused-ignore]

__all__ = ["WhisperModel"]

class WhisperModel(PretrainedModelMixin):
    """
    A base model for whisper models.
    """
    @classmethod
    def get_model_class(cls) -> Type[WhisperForConditionalGeneration]:
        """
        Returns the model class.
        """
        from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration # type: ignore[import-not-found,import-untyped,unused-ignore]
        return WhisperForConditionalGeneration # type: ignore[no-any-return]

    @classmethod
    def get_config_class(cls) -> Type[WhisperConfig]:
        """
        Returns the configuration class.
        """
        from transformers.models.whisper.configuration_whisper import WhisperConfig # type: ignore[import-not-found,import-untyped,unused-ignore]
        return WhisperConfig # type: ignore[no-any-return]
