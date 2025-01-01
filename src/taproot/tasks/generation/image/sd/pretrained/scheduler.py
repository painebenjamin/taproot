from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers import DDIMScheduler

__all__ = ["StableDiffusionScheduler"]

class StableDiffusionScheduler(PretrainedModelMixin):
    """
    The Stable Diffusion Scheduler.
    """
    @classmethod
    def get_model_class(cls) -> Type[DDIMScheduler]:
        """
        Get the model class for the Stable Diffusion Scheduler.
        """
        from diffusers import DDIMScheduler
        return DDIMScheduler

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Stable Diffusion Scheduler.
        """
        return {
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "num_train_timesteps": 1000,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None,
            "clip_sample": False
        }
