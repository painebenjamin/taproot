from __future__ import annotations

from inspect import signature

from typing import Dict, Any, Optional, List, TYPE_CHECKING

from .availability_util import diffusers_is_available
from ...constants import *

if TYPE_CHECKING:
    from diffusers.schedulers.scheduling_utils import SchedulerMixin

__all__ = [
    "get_diffusers_scheduler_by_name",
    "get_aligned_timesteps_for_scheduler",
    "scheduler_accepts_timesteps",
]

def scheduler_accepts_timesteps(scheduler: SchedulerMixin) -> bool:
    """
    Check if the scheduler accepts timesteps in the `set_timesteps` method.
    """
    method = getattr(scheduler, "set_timesteps", None)
    if method is None:
        return False
    return "timesteps" in set(signature(method).parameters.keys())

def get_aligned_timesteps_for_scheduler(
    scheduler: SchedulerMixin,
    model_type: DIFFUSERS_MODEL_TYPE_LITERAL,
    num_timesteps: Optional[int]=None
) -> Optional[List[int]]:
    """
    Get the aligned timesteps for the scheduler.
    """
    if not scheduler_accepts_timesteps(scheduler):
        return None
    elif hasattr(scheduler, "config") and getattr(scheduler.config, "use_karras_sigmas", False):
        return None

    from diffusers.schedulers.scheduling_utils import AysSchedules

    schedule: List[int] = []
    if model_type.lower() in ["sdxl", "stable-diffusion-xl"]:
        schedule = AysSchedules["StableDiffusionXLTimesteps"] # type: ignore[assignment]
    elif model_type.lower() in ["sd", "stable-diffusion", "stable-diffusion-v1-5"]:
        schedule = AysSchedules["StableDiffusionTimesteps"] # type: ignore[assignment]
    else:
        return None

    if num_timesteps is None or num_timesteps == len(schedule):
        return schedule

    # Do log-linear interpolation
    from numpy import log, exp, linspace, interp
    x_vals = linspace(0, 1, len(schedule))
    y_vals = log(schedule[::-1])
    new_x_vals = linspace(0, 1, num_timesteps)
    new_y_vals = interp(new_x_vals, x_vals, y_vals)
    return exp(new_y_vals)[::-1].astype(int).tolist() # type: ignore[no-any-return]

def get_diffusers_scheduler_by_name(
    name: DIFFUSERS_SCHEDULER_LITERAL,
    config: Optional[Dict[str, Any]]=None
) -> SchedulerMixin:
    """
    Get the scheduler class by name.
    """
    if not diffusers_is_available():
        raise ImportError("Diffusers is not available. Configure your task to require it if needed.")
    use_karras_sigmas = name.endswith("_karras")
    if config is not None:
        config["use_karras_sigmas"] = use_karras_sigmas
    if name == "ddim":
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        if config is None:
            return DDIMScheduler() # type: ignore[no-any-return]
        return DDIMScheduler.from_config(config) # type: ignore[no-any-return]
    elif name == "ddpm":
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        if config is None:
            return DDPMScheduler() # type: ignore[no-any-return]
        return DDPMScheduler.from_config(config) # type: ignore[no-any-return]
    elif name == "ddpm_wuerstchen":
        from diffusers.schedulers.scheduling_ddpm_wuerstchen import DDPMWuerstchenScheduler
        if config is None:
            return DDPMWuerstchenScheduler() # type: ignore[no-any-return]
        return DDPMWuerstchenScheduler.from_config(config) # type: ignore[no-any-return]
    elif name == "deis_multistep":
        from diffusers.schedulers.scheduling_deis_multistep import DEISMultistepScheduler
        if config is None:
            return DEISMultistepScheduler() # type: ignore[no-any-return]
        return DEISMultistepScheduler.from_config(config) # type: ignore[no-any-return]
    elif name == "dpm_cogvideox":
        from diffusers.schedulers.scheduling_dpm_cogvideox import CogVideoXDPMScheduler
        if config is None:
            return CogVideoXDPMScheduler() # type: ignore[no-any-return]
        return CogVideoXDPMScheduler.from_config(config) # type: ignore[no-any-return]
    elif name == "dpmsolver_sde":
        from diffusers.schedulers.scheduling_dpmsolver_sde import DPMSolverSDEScheduler
        if config is None:
            return DPMSolverSDEScheduler() # type: ignore[no-any-return]
        return DPMSolverSDEScheduler.from_config(config) # type: ignore[no-any-return]
    elif name in ["dpmsolver_multistep", "dpmsolver_multistep_karras"]:
        from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        if config is None:
            return DPMSolverMultistepScheduler(use_karras_sigmas=use_karras_sigmas) # type: ignore[no-any-return]
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=use_karras_sigmas) # type: ignore[no-any-return]
    elif name in ["dpmsolver_sde_multistep", "dpmsolver_sde_multistep_karras"]:
        from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        if config is None:
            return DPMSolverMultistepScheduler( # type: ignore[no-any-return]
                use_karras_sigmas=use_karras_sigmas,
                algorithm_type="sde-dpmsolver++"
            )
        return DPMSolverMultistepScheduler.from_config( # type: ignore[no-any-return]
            config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=use_karras_sigmas
        )
    elif name in ["dpmsolver_singlestep", "dpmsolver_singlestep_karras"]:
        from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
        if config is None:
            return DPMSolverSinglestepScheduler(use_karras_sigmas=use_karras_sigmas) # type: ignore[no-any-return]
        return DPMSolverSinglestepScheduler.from_config(config, use_karras_sigmas=use_karras_sigmas) # type: ignore[no-any-return]
    elif name == "edm_dpmsolver_multistep":
        from diffusers.schedulers.scheduling_edm_dpmsolver_multistep import EDMDPMSolverMultistepScheduler
        if config is None:
            return EDMDPMSolverMultistepScheduler() # type: ignore[no-any-return]
        return EDMDPMSolverMultistepScheduler.from_config(config) # type: ignore[no-any-return]
    elif name == "edm_euler":
        from diffusers.schedulers.scheduling_edm_euler import EDMEulerScheduler
        if config is None:
            return EDMEulerScheduler() # type: ignore[no-any-return]
        return EDMEulerScheduler.from_config(config) # type: ignore[no-any-return]
    elif name == "euler_ancestral_discrete":
        from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
        if config is None:
            return EulerAncestralDiscreteScheduler() # type: ignore[no-any-return]
        return EulerAncestralDiscreteScheduler.from_config(config) # type: ignore[no-any-return]
    elif name in ["euler_discrete", "euler_discrete_karras"]:
        from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
        if config is None:
            return EulerDiscreteScheduler(use_karras_sigmas=use_karras_sigmas) # type: ignore[no-any-return]
        return EulerDiscreteScheduler.from_config(config, use_karras_sigmas=use_karras_sigmas) # type: ignore[no-any-return]
    elif name == "flow_match_euler_discrete":
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        if config is None:
            return FlowMatchEulerDiscreteScheduler() # type: ignore[no-any-return]
        return FlowMatchEulerDiscreteScheduler.from_config(config) # type: ignore[no-any-return]
    elif name == "flow_match_heun_discrete":
        from diffusers.schedulers.scheduling_flow_match_heun_discrete import FlowMatchHeunDiscreteScheduler
        if config is None:
            return FlowMatchHeunDiscreteScheduler() # type: ignore[no-any-return]
        return FlowMatchHeunDiscreteScheduler.from_config(config) # type: ignore[no-any-return]
    elif name == "heun_discrete":
        from diffusers.schedulers.scheduling_heun_discrete import HeunDiscreteScheduler
        if config is None:
            return HeunDiscreteScheduler() # type: ignore[no-any-return]
        return HeunDiscreteScheduler.from_config(config) # type: ignore[no-any-return]
    elif name == "ipndm":
        from diffusers.schedulers.scheduling_ipndm import IPNDMScheduler
        if config is None:
            return IPNDMScheduler() # type: ignore[no-any-return]
        return IPNDMScheduler.from_config(config) # type: ignore[no-any-return]
    elif name in ["k_dpm_2_ancestral_discrete", "k_dpm_2_ancestral_discrete_karras"]:
        from diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete import KDPM2AncestralDiscreteScheduler
        if config is None:
            return KDPM2AncestralDiscreteScheduler(use_karras_sigmas=use_karras_sigmas) # type: ignore[no-any-return]
        return KDPM2AncestralDiscreteScheduler.from_config(config, use_karras_sigmas=use_karras_sigmas) # type: ignore[no-any-return]
    elif name in ["k_dpm_2_discrete", "k_dpm_2_discrete_karras"]:
        from diffusers.schedulers.scheduling_k_dpm_2_discrete import KDPM2DiscreteScheduler
        if config is None:
            return KDPM2DiscreteScheduler(use_karras_sigmas=use_karras_sigmas) # type: ignore[no-any-return]
        return KDPM2DiscreteScheduler.from_config(config, use_karras_sigmas=use_karras_sigmas) # type: ignore[no-any-return]
    elif name == "lcm":
        from diffusers.schedulers.scheduling_lcm import LCMScheduler
        if config is None:
            return LCMScheduler() # type: ignore[no-any-return]
        return LCMScheduler.from_config(config) # type: ignore[no-any-return]
    elif name in ["lms_discrete", "lms_discrete_karras"]:
        from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
        if config is None:
            return LMSDiscreteScheduler(use_karras_sigmas=use_karras_sigmas) # type: ignore[no-any-return]
        return LMSDiscreteScheduler.from_config(config, use_karras_sigmas=use_karras_sigmas) # type: ignore[no-any-return]
    elif name == "pndm":
        from diffusers.schedulers.scheduling_pndm import PNDMScheduler
        if config is None:
            return PNDMScheduler() # type: ignore[no-any-return]
        return PNDMScheduler.from_config(config) # type: ignore[no-any-return]
    elif name == "tcd":
        from diffusers.schedulers.scheduling_tcd import TCDScheduler
        if config is None:
            return TCDScheduler() # type: ignore[no-any-return]
        return TCDScheduler.from_config(config) # type: ignore[no-any-return]
    elif name == "unipc":
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        if config is None:
            return UniPCMultistepScheduler() # type: ignore[no-any-return]
        return UniPCMultistepScheduler.from_config(config) # type: ignore[no-any-return]
    else:
        raise ValueError(f"Invalid scheduler name: {name}")
