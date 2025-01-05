from __future__ import annotations

from typing import Any, Dict, Optional, Type, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from .model import KokoroModel

__all__ = ["KokoroV019Model"]

class KokoroV019Model(PretrainedModelMixin):
    """
    Pretrained Kokoro V0.19 Model.
    """
    model_url = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/speech-synthesis-kokoro-v0-19.safetensors"
    init_classmethod = "from_config"
    init_file_urls = {
        "voices": "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/speech-synthesis-kokoro-v0-19-voices.safetensors",
    }

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        return {
            "decoder": {
                "in_dim": 512,
                "style_dim": 128,
                "out_dim": 80, # mel spectrogram
                "upsample_kernel_sizes": (20, 12),
                "upsample_rates": (10, 6),
                "gen_istft_hop_size": 5,
                "gen_istft_n_fft": 20,
                "resblock_dilation_sizes": (
                    (1, 3, 5),
                    (1, 3, 5),
                    (1, 3, 5),
                ),
                "resblock_kernel_sizes": (3, 7, 11),
                "upsample_initial_channel": 512
            },
            "text_encoder": {
                "channels": 512,
                "kernel_size": 5,
                "num_layers": 3,
                "num_symbols": 178,
            },
            "predictor": {
                "style_dim": 128,
                "hidden_dim": 512,
                "num_layers": 3,
                "max_duration": 50,
                "dropout": 0.2,
            }
        }

    @classmethod
    def get_model_class(cls) -> Type[KokoroModel]:
        """
        Returns the model class.
        """
        from .model import KokoroModel
        return KokoroModel
