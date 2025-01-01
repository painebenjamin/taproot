from __future__ import annotations

import os
import json

from typing import Optional, Type, Dict, Any, Union

from omegaconf import OmegaConf
from omegaconf.basecontainer import BaseContainer

__all__ = ["ConfigMixin", "ConfigType"]

ConfigType = Optional[Union[str, Dict[str, Any], BaseContainer]]

class ConfigMixin:
    """
    Mixin class to get/set structured configuration data
    """
    config_class: Optional[Type[Any]] = None

    def __init__(self, config: ConfigType = None) -> None:
        """
        Initialize the configuration data
        """
        if self.config_class is None:
            self.config = OmegaConf.create()
        else:
            self.config = OmegaConf.structured(self.config_class)

        if config is not None:
            if isinstance(config, str):
                _, ext = os.path.splitext(config)
                if ext in [".yml", ".yaml"]:
                    self.config.merge_with(OmegaConf.load(config))
                elif ext == ".json":
                    with open(config, "r") as f:
                        self.config.merge_with(OmegaConf.create(json.load(f)))
                else:
                    raise ValueError(f"Unknown file extension for configuration: {ext}")
            elif isinstance(config, dict):
                self.config.merge_with(OmegaConf.create(config))
            elif isinstance(config, BaseContainer):
                self.config.merge_with(config)
            else:
                raise ValueError(f"Unknown configuration type: {type(config)}")
