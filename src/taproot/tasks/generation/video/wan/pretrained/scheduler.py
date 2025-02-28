from __future__ import annotations

from typing import Type, Optional, Dict, Any, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from ..schedulers import FlowMatchUniPCMultistepScheduler # type: ignore[attr-defined]

__all__ = ["PretrainedWanScheduler"]

class PretrainedWanScheduler(PretrainedModelMixin):
    """
    The Wan Video Scheduler.
    """
    @classmethod
    def get_model_class(cls) -> Type[FlowMatchUniPCMultistepScheduler]:
        """
        Get the model class for the Hunyuan Video Scheduler.
        """
        from ..schedulers import FlowMatchUniPCMultistepScheduler # type: ignore[attr-defined]
        return FlowMatchUniPCMultistepScheduler # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration for the Hunyuan Video Scheduler.
        """
        return {
            "shift": 1,
            "use_dynamic_shifting": False,
            "num_train_timesteps": 1000,
        }
