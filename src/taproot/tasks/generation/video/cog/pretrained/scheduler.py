from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from diffusers.schedulers import CogVideoXDDIMScheduler

__all__ = [
    "CogVideoX2BScheduler",
    "CogVideoX5BScheduler"
]

class CogVideoXScheduler(PretrainedModelMixin):
    """
    The CogVideoX Scheduler.
    """
    @classmethod
    def get_model_class(cls) -> Type[CogVideoXDDIMScheduler]:
        """
        Get the model class for the CogVideoX Scheduler.
        """
        from diffusers.schedulers import CogVideoXDDIMScheduler
        return CogVideoXDDIMScheduler

class CogVideoX2BScheduler(CogVideoXScheduler):
    """
    The CogVideoX Scheduler for the CogVideoX 2B model.
    """
    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the CogVideoX 2B Scheduler.
        """
        return {
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "clip_sample_range": 1.0,
            "num_train_timesteps": 1000,
            "prediction_type": "v_prediction",
            "rescale_betas_zero_snr": True,
            "sample_max_value": 1.0,
            "set_alpha_to_one": True,
            "snr_shift_scale": 3.0,
            "steps_offset": 0,
            "timestep_spacing": "trailing",
            "trained_betas": None
        }

class CogVideoX5BScheduler(CogVideoXScheduler):
    """
    The CogVideoX Scheduler for the CogVideoX 5B models.
    """
    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the CogVideoX 5B Scheduler.
        """
        return {
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "clip_sample_range": 1.0,
            "num_train_timesteps": 1000,
            "prediction_type": "v_prediction",
            "rescale_betas_zero_snr": True,
            "sample_max_value": 1.0,
            "set_alpha_to_one": True,
            "snr_shift_scale": 1.0,
            "steps_offset": 0,
            "timestep_spacing": "trailing",
            "trained_betas": None
        }
