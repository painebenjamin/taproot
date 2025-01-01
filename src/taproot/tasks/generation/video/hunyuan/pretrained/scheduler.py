from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

__all__ = ["HunyuanVideoScheduler"]

class HunyuanVideoScheduler(PretrainedModelMixin):
    """
    The Hunyuan Video Scheduler.
    """

    @classmethod
    def get_model_class(cls) -> Type[FlowMatchEulerDiscreteScheduler]:
        """
        Get the model class for the Hunyuan Video Scheduler.
        """
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        return FlowMatchEulerDiscreteScheduler

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Hunyuan Video Scheduler.
        """
        return {
            "base_image_seq_len": 256,
            "base_shift": 0.5,
            "invert_sigmas": False,
            "max_image_seq_len": 4096,
            "max_shift": 1.15,
            "num_train_timesteps": 1000,
            "shift": 7.0,
            "shift_terminal": None,
            "use_beta_sigmas": False,
            "use_dynamic_shifting": False,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False
        }
