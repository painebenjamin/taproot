from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

__all__ = ["MochiScheduler"]

class MochiScheduler(PretrainedModelMixin):
    """
    The Mochi Scheduler.
    """

    @classmethod
    def get_model_class(cls) -> Type[FlowMatchEulerDiscreteScheduler]:
        """
        Get the model class for the LTX Video Scheduler.
        """
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        return FlowMatchEulerDiscreteScheduler

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the LTX Video Scheduler.
        """
        return {
            "base_image_seq_len": 256,
            "base_shift": 0.5,
            "invert_sigmas": True,
            "max_image_seq_len": 4096,
            "max_shift": 1.15,
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "use_dynamic_shifting": False
        }
