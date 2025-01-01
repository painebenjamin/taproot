from typing import Any, Dict, Optional

from ..pretrained import SDXLScheduler

__all__ = ["SDXLTurboScheduler"]

class SDXLTurboScheduler(SDXLScheduler):
    """
    Scheduler for SDXL models using Turbo.
    Default scheduler model is the same for base and turbo,
    but there is a small configuration override.
    """
    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Override parent config to set timestep spacing.
        """
        config = super().get_default_config()
        assert isinstance(config, dict), "base config must be a dictionary"
        config["timestep_spacing"] = "trailing"
        return config
