from __future__ import annotations

import os

from typing import Optional, Union, List, Dict, TYPE_CHECKING
from typing_extensions import TypedDict

from ..log_util import logger

if TYPE_CHECKING:
    import torch
    from optimum.quanto.tensor import Optimizer, qtype # type: ignore[import-untyped]

class QuantizationMap(TypedDict):
    weights: str # qtype or 'none'
    activations: str # qtype or 'none'

class OptimumQuantoQuantizer:
    """
    A class that encapsulates the quantization configuration for a model using Optimum-Quanto.

    Only modules that have quantized counterparts will be quantized.

    If include patterns are specified, the submodule name must match one of them.
    If exclude patterns are specified, the submodule must not match one of them.

    Include or exclude patterns are Unix shell-style wildcards which are NOT regular expressions. See
    https://docs.python.org/3/library/fnmatch.html for more details.
    """
    def __init__(
        self,
        pre_quantized: bool=False,
        weights: Optional[Union[str, torch.dtype, qtype]]=None,
        activations: Optional[Union[str, torch.dtype, qtype]]=None,
        optimizer: Optional[Optimizer]=None,
        include: Optional[Union[str, List[str]]]=None,
        exclude: Optional[Union[str, List[str]]]=None,
    ) -> None:
        """
        :param pre_quantized: Whether the model is pre-quantized.
        :param weights: The quantization type for weights.
        :param activations: The quantization type for activations.
        :param optimizer: The optimizer to use for quantization.
        :param include: Patterns constituting the allowlist. If provided, module names must match at least one pattern from the allowlist.
        :param exclude: Patterns constituting the denylist. If provided, module names must not match any patterns from the denylist.
        """
        import torch
        from optimum.quanto.tensor.qtype import qtypes, qfloat # type: ignore[import-untyped]
        if isinstance(weights, str):
            weights = qtypes[weights]
        elif isinstance(weights, torch.dtype):
            weights = qfloat(weights)
        self.weights = weights
        if isinstance(activations, str):
            activations = qtypes[activations]
        elif isinstance(activations, torch.dtype):
            activations = qfloat(activations)
        self.activations = activations
        self.optimizer = optimizer
        self.pre_quantized = pre_quantized
        self.include = include
        self.exclude = exclude

    @property
    def default_quantization_map(self) -> QuantizationMap:
        """
        The default quantization map for the model.
        """
        return {
            "weights": "none" if self.weights is None else self.weights.name,
            "activations": "none" if self.activations is None else self.activations.name
        }

    def set_module_by_name(
        self,
        parent_module: torch.nn.Module,
        name: str,
        child_module: torch.nn.Module
    ) -> None:
        """
        Set a module in a parent module by name.

        :param parent_module: The parent module.
        :param name: The name of the module to set.
        :param child_module: The module to set.
        :see https://github.com/huggingface/optimum-quanto/blob/main/optimum/quanto/quantize.py:
        """
        module_names = name.split(".")
        if len(module_names) == 1:
            setattr(parent_module, name, child_module)
        else:
            parent_module_name = name[: name.rindex(".")]
            parent_module = parent_module.get_submodule(parent_module_name)
            setattr(parent_module, module_names[-1], child_module)

    def quantize_submodule(
        self,
        model: torch.nn.Module,
        name: str,
        module: torch.nn.Module,
        weights: Optional[qtype]=None,
        activations: Optional[qtype]=None,
        optimizer: Optional[Optimizer]=None,
    ) -> None:
        """
        Quantize a submodule of a model.

        :param model: The model containing the submodule.
        :param name: The name of the submodule.
        :param module: The submodule to quantize.
        :param weights: The quantization type for weights.
        :param activations: The quantization type for activations.
        :param optimizer: The optimizer to use for quantization.
        """
        from optimum.quanto.nn import quantize_module # type: ignore[import-untyped]
        qmodule = quantize_module(
            module,
            weights=weights,
            activations=activations,
            optimizer=optimizer
        )
        if qmodule is not None:
            self.set_module_by_name(model, name, qmodule)
            qmodule.name = name
            for name, param in module.named_parameters():
                # Save device memory by clearing parameters
                setattr(module, name, None)
                del param

    def quantize(self, model: torch.nn.Module) -> None:
        """
        Quantize the specified model submodules

        Recursively quantize the submodules of the specified parent model.

        Note: quantization happens in-place and modifies the original model and its descendants.

        :param model: the model whose submodules will be quantized.
        """
        from fnmatch import fnmatch
        include = self.include
        exclude = self.exclude
        if include is not None:
            include = [include] if isinstance(include, str) else include
        if exclude is not None:
            exclude = [exclude] if isinstance(exclude, str) else exclude
        for name, m in model.named_modules():
            if include is not None and not any(fnmatch(name, pattern) for pattern in include):
                continue
            if exclude is not None and any(fnmatch(name, pattern) for pattern in exclude):
                continue
            self.quantize_submodule(
                model=model,
                name=name,
                module=m,
                weights=self.weights,
                activations=self.activations,
                optimizer=self.optimizer
            )

    def freeze(self, model: torch.nn.Module) -> None:
        """
        Freeze the quantized modules in the model.
        """
        from optimum.quanto.nn import QModuleMixin
        for name, m in model.named_modules():
            if isinstance(m, QModuleMixin):
                m.freeze()

    def guess_quantization_map(
        self,
        model: torch.nn.Module
    ) -> Dict[str, QuantizationMap]:
        """
        Given a model, return a quantization map that would be
        generated when quantizing the model. This assumes the
        model has _not_ been quantized yet, unlike the default
        `get_quantization_map` method.
        """
        from fnmatch import fnmatch
        quantization_map: Dict[str, QuantizationMap] = {}
        for name, m in model.named_modules():
            if self.include is not None and not any(fnmatch(name, pattern) for pattern in self.include):
                continue
            if self.exclude is not None and any(fnmatch(name, pattern) for pattern in self.exclude):
                continue
            qconfig = self.guess_submodule_quantization_map(m)
            if qconfig is not None:
                quantization_map[name] = qconfig
            elif m is not model:
                sub_quantization_map = self.guess_quantization_map(m)
                for key, value in sub_quantization_map.items():
                    quantization_map[f"{name}.{key}"] = value
        return quantization_map

    def guess_submodule_quantization_map(
        self,
        model: torch.nn.Module,
    ) -> Optional[QuantizationMap]:
        """
        Given a model, return a quantization map that would be
        generated when quantizing the model. This assumes the
        model has _not_ been quantized yet, unlike the default
        `get_quantization_map` method.
        """
        from optimum.quanto.nn.qmodule import _QMODULE_TABLE # type: ignore[import-untyped]
        for cls in _QMODULE_TABLE:
            if isinstance(model, cls):
                return self.default_quantization_map
        return None

    def guess_quantization_map_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, QuantizationMap]:
        """
        Given a state_dict, return a quantization map that specifies the quantization type for each tensor in the state_dict.

        :param state_dict: The state_dict of the model.
        :return: A quantization map that specifies the quantization type for each tensor in the state_dict.
        """
        if self.weights is None:
            return {} # Cannot guess quantization map without knowing the quantization type of weights
        quantization_map: Dict[str, QuantizationMap] = {}
        for key, value in state_dict.items():
            if value.dtype is self.weights.dtype:
                quantization_map[key] = {
                    "weights": self.weights.name,
                    "activations": "none" if self.activations is None else self.activations.name
                }

        return quantization_map

    def requantize(
        self,
        model: torch.nn.Module,
        checkpoint: Union[str, List[str]],
        quantization_map: Optional[Dict[str, QuantizationMap]]=None,
        device: Optional[torch.device]=None
    ) -> None:
        """
        Requantize the model using the specified checkpoint(s).

        :param model: The model to requantize.
        :param checkpoint: The checkpoint(s) to use for requantization.
        :param quantization_map: The quantization map to use for requantization.
        :param device: The device to use for requantization.
        """
        import torch
        import safetensors

        checkpoints = [checkpoint] if isinstance(checkpoint, str) else checkpoint

        if device is None:
            device = next(model.parameters()).device
            if device.type == "meta":
                device = torch.device("cpu")

        # Get the quantization map if not provided
        if quantization_map is None:
            quantization_map = self.guess_quantization_map(model)
            logger.debug(f"Guessed quantization map: {quantization_map}")

        # Quantize the model with parameters from the quantization map
        for name, m in model.named_modules():
            qconfig = quantization_map.get(name, None)
            if qconfig is not None:
                weights: Optional[str] = qconfig["weights"]
                activations: Optional[str] = qconfig["activations"]
                if weights == "none":
                    weights = None
                if activations == "none":
                    activations = None

                self.quantize_submodule(
                    model=model,
                    name=name,
                    module=m,
                    weights=weights,
                    activations=activations
                )

        # Move model parameters and buffers to CPU before materializing quantized weights
        for name, m in model.named_modules():
            # Helper function to move tensors to the target device
            def move_tensor(t: torch.Tensor, device: str) -> torch.Tensor:
                if t.device.type == "meta":
                    return torch.empty_like(t, device=device)
                return t.to(device)

            for name, param in m.named_parameters(recurse=False):
                setattr(m, name, torch.nn.Parameter(move_tensor(param, "cpu")))
            for name, param in m.named_buffers(recurse=False):
                setattr(m, name, move_tensor(param, "cpu"))

        # Move to target device
        model.to(device)
        # Load the quantized model weights
        for checkpoint in checkpoints:
            _, ext = os.path.splitext(checkpoint)
            if ext == ".safetensors":
                safetensors.torch.load_model(model, checkpoint)
            else:
                model.load_state_dict(torch.load(checkpoint, map_location=device))

    def postprocess_model(self, model: torch.nn.Module) -> None:
        """
        Postprocess the model after quantization or requantization.

        :param model: The model to postprocess.
        """
        if not self.pre_quantized:
            # When the model is not pre-quantized, quantization happens
            # after initial state_dict loading
            logger.debug(f"Quantizing model {model.__class__.__name__} with weights={self.weights}, activations={self.activations}, optimizer={self.optimizer}")
            self.quantize(model)
        logger.debug(f"Freezing model {model.__class__.__name__}")
        self.freeze(model)
