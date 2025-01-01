from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers import EulerDiscreteScheduler

__all__ = ["SDXLScheduler"]

class SDXLScheduler(PretrainedModelMixin):
    """
    The Stable Diffusion XL Scheduler.
    """
    @classmethod
    def get_model_class(cls) -> Type[EulerDiscreteScheduler]:
        """
        Get the model class for the Stable Diffusion XL Scheduler.
        """
        from diffusers import EulerDiscreteScheduler
        return EulerDiscreteScheduler

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Stable Diffusion XL Scheduler.
        """
        return {
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "interpolation_type": "linear",
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "sample_max_value": 1,
            "set_alpha_to_one": False,
            "skip_prk_steps":True,
            "steps_offset": 1,
            "timestep_spacing": "leading",
            "trained_betas": None,
            "use_karras_sigmas": False
        }
