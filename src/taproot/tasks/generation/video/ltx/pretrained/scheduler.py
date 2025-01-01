from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

__all__ = ["LTXVideoScheduler"]

class LTXVideoScheduler(PretrainedModelMixin):
    """
    The LTX Video Scheduler.
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
            "base_image_seq_len": 1024,
            "base_shift": 0.95,
            "invert_sigmas": False,
            "max_image_seq_len": 4096,
            "max_shift": 2.05,
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": 0.1,
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False
        }
