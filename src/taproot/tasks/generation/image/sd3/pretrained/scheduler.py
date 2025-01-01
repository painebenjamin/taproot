from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

__all__ = ["StableDiffusion3Scheduler"]

class StableDiffusion3Scheduler(PretrainedModelMixin):
    """
    The Stable Diffusion 3 Scheduler.
    """
    @classmethod
    def get_model_class(cls) -> Type[FlowMatchEulerDiscreteScheduler]:
        """
        Get the model class for the Stable Diffusion 3 Scheduler.
        """
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        return FlowMatchEulerDiscreteScheduler

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Stable Diffusion 3 Scheduler.
        """
        return {
            "num_train_timesteps": 1000,
            "shift": 3.0
        }