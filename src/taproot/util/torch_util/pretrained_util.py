from __future__ import annotations

import os
from typing import Any, Optional, Type, Union, Dict, List, Tuple, Sequence, TYPE_CHECKING
from omegaconf import OmegaConf
from contextlib import nullcontext

from .quantization_util import OptimumQuantoQuantizer

if TYPE_CHECKING:
    import torch
    from transformers.configuration_utils import PretrainedConfig # type: ignore[import-not-found,import-untyped,unused-ignore]
    from transformers.quantizers import AutoHfQuantizer as TransformersAutoQuantizer # type: ignore[import-not-found,import-untyped,unused-ignore]
    from diffusers.quantizers import DiffusersAutoQuantizer # type: ignore[attr-defined]

__all__ = ["PretrainedModelMixin"]

class PretrainedModelMixin:
    """
    A mixin class for diffusers, transformers, or regular torch models.
    Provides utilities for downloading and loading pretrained models.
    """
    model_url: Optional[Union[str, List[str]]] = None
    init_file_urls: Optional[Dict[str, Union[str, List[str]]]] = None
    quantization: Optional[str] = None
    pre_quantized: bool = False
    load_path: Optional[str] = None
    use_eval: bool = True
    use_strict: bool = True
    tie_weights: bool = True
    use_compile: bool = False
    spread_config: bool = True
    use_accelerate: bool = True
    load_on_device: bool = True
    no_init_weights: bool = True
    move_after_load: bool = True
    use_torch_jit: bool = False
    dtype: Optional[Union[str, torch.dtype]] = None # Override dtype for the model

    @classmethod
    def get_required_files(cls) -> List[str]:
        """
        Get the required files for the model.
        """
        required_files: List[str] = []
        if cls.model_url is not None:
            if isinstance(cls.model_url, str):
                required_files.append(cls.model_url)
            else:
                required_files.extend(cls.model_url)
        if cls.init_file_urls is not None:
            for init_files in cls.init_file_urls.values():
                if isinstance(init_files, str):
                    required_files.append(init_files)
                else:
                    required_files.extend(init_files)
        return required_files

    @classmethod
    def get_model_class(cls) -> Optional[Type[Any]]:
        """
        Get the model class. Default is none.
        """
        return None

    @classmethod
    def get_config_class(cls) -> Optional[Type[Any]]:
        """
        Get the configuration class. Default is none.
        """
        return None

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration. Default is none.
        """
        return None

    @classmethod
    def get_generation_config(cls) -> Optional[Dict[str, Any]]:
        """
        Get the generation configuration. Default is none.
        """
        return None

    @classmethod
    def get_quantization_config(cls) -> Any:
        """
        Gets quantization configuration.
        """
        if cls.quantization is None:
            return None

        if cls.quantization.startswith("bitsandbytes"):
            if not cls.is_bitsandbytes_available():
                return None
            import torch
            if cls.is_diffusers_model():
                from diffusers import BitsAndBytesConfig # type: ignore[attr-defined,import-not-found,unused-ignore]
            elif cls.is_transformers_model():
                from transformers import BitsAndBytesConfig # type: ignore[import-untyped,import-not-found,unused-ignore,no-redef]
            else:
                raise ValueError("Quantization is only supported for diffusers and transformers models.")
            kwargs: Dict[str, Any] = {}
            if cls.quantization == "bitsandbytes_4bit" or cls.quantization == "bitsandbytes_nf4":
                kwargs["load_in_4bit"] = True
                if cls.quantization == "bitsandbytes_nf4":
                    kwargs["bnb_4bit_quant_type"] = "nf4"
                    kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
            elif cls.quantization == "bitsandbytes_8bit":
                kwargs["load_in_8bit"] = True
            else:
                raise ValueError(f"Invalid quantization method {cls.quantization}.")
            return BitsAndBytesConfig(**kwargs) # type: ignore[no-untyped-call]
        elif cls.quantization.startswith("quanto"):
            if not cls.is_optimum_quanto_available():
                return None
            _, _, qtype = cls.quantization.partition("_")
            return {
                "method": "quanto",
                "qtype": qtype
            }
        else:
            raise ValueError(f"Invalid quantization method {cls.quantization}.")

    @classmethod
    def get_load_target(cls, model: Any) -> torch.nn.Module:
        """
        Get the target model to load the state dict into.
        """
        import torch
        if cls.load_path is None:
            assert isinstance(model, torch.nn.Module), "Model must be an NN module if load path is empty."
            return model

        # Now lex the load path
        target = model
        load_chunks = cls.load_path.split(".")
        for chunk in load_chunks:
            # Check for a numeric chunk to index into the target
            if chunk.isdigit():
                target = target[int(chunk)]
            else:
                target = getattr(target, chunk, None)
                assert target is not None, f"Could not find attribute {chunk} in {model}."
        assert isinstance(target, torch.nn.Module), "Target must be an NN module, check the load path."
        return target

    @classmethod
    def post_load_hook(cls, model: Any) -> None:
        """
        Post-load hook for the model.
        """
        pass

    @classmethod
    def is_accelerate_available(cls) -> bool:
        """
        Check if the accelerate library is available.
        """
        try:
            import accelerate # type: ignore[import-not-found,import-untyped,unused-ignore]
            return accelerate is not None
        except ImportError:
            from ..log_util import logger
            logger.warning("Accelerate library requested but not available. Use `pip install accelerate` to install and speed up module loading.")
            return False

    @classmethod
    def is_bitsandbytes_available(cls) -> bool:
        """
        Check if the bitsandbytes library is available.
        """
        try:
            import bitsandbytes # type: ignore[import-not-found,import-untyped,unused-ignore]
            return bitsandbytes is not None
        except ImportError:
            from ..log_util import logger
            logger.warning("BitsAndBytes library requested but not available. Use `pip install bitsandbytes` to install and reduce model size.")
            return False

    @classmethod
    def is_optimum_quanto_available(cls) -> bool:
        """
        Check if the optimum-quanto library is available.
        """
        try:
            from optimum import quanto # type: ignore[import-not-found,import-untyped,unused-ignore]
            return quanto is not None
        except ImportError:
            from ..log_util import logger
            logger.warning("Optimum-Quanto library requested but not available. Use `pip install optimum-quanto` to install and reduce model size.")
            return False

    @classmethod
    def merge_default_config(
        cls,
        config: Union[PretrainedConfig, OmegaConf, Dict[str, Any]]
    ) -> Union[PretrainedConfig, OmegaConf, Dict[str, Any]]:
        """
        Merge the default configuration with the provided configuration.
        """
        default_config = cls.get_default_config()
        if default_config is not None:
            if isinstance(config, dict):
                from ..misc_util import merge_into
                return merge_into(config, default_config)
            elif isinstance(config, OmegaConf):
                return OmegaConf.merge(default_config, config)
            elif cls.is_transformers_pretrained_config(config):
                config.update(default_config)
            return config
        return config

    @classmethod
    def is_transformers_model(cls) -> bool:
        """
        Check if the model is a transformers model.
        """
        model_class = cls.get_model_class()
        if model_class is not None:
            return "PreTrainedModel" in [c.__name__ for c in model_class.__mro__] # Note capitalization
        return False

    @classmethod
    def is_transformers_pretrained_config(cls, config_class: Optional[Type[PretrainedConfig]]=None) -> bool:
        """
        Check if the configuration is a transformers configuration.
        """
        config_class = cls.get_config_class() if config_class is None else config_class
        if config_class is not None:
            return "PretrainedConfig" in [c.__name__ for c in config_class.__mro__] # Note capitalization
        return False

    @classmethod
    def is_diffusers_model(cls) -> bool:
        """
        Check if the model is a diffusers model.
        """
        parent_classes = ["ModelMixin", "SchedulerMixin"]
        model_class = cls.get_model_class()
        if model_class is not None:
            return any([c.__name__ in parent_classes for c in model_class.__mro__])
        return False

    @classmethod
    def instantiate_transformers_model(cls, **kwargs: Any) -> Any:
        """
        Instantiate a transformers model.
        """
        from ..log_util import logger
        model_class = cls.get_model_class()
        assert model_class is not None, "Model class is not defined, this method should not be called."
        config_class = cls.get_config_class()
        config = cls.merge_default_config(kwargs)
        if config_class is None:
            from transformers.configuration_utils import PretrainedConfig
            config_class = PretrainedConfig
        assert hasattr(config_class, "from_dict"), "Configuration class must have a from_dict method."
        model = model_class(config_class.from_dict(config))
        generation_config = cls.get_generation_config()
        if generation_config is not None:
            from transformers.generation.configuration_utils import GenerationConfig # type: ignore[import-not-found,import-untyped,unused-ignore]
            model.generation_config = GenerationConfig(**generation_config)
        return model

    @classmethod
    def instantiate_diffusers_model(cls, **kwargs: Any) -> Any:
        """
        Instantiate a diffusers model.
        """
        model_class = cls.get_model_class()
        assert model_class is not None, "Model class is not defined, this method should not be called."
        config = cls.merge_default_config(kwargs)
        assert isinstance(config, dict), "Configuration must be a dictionary."
        return model_class.from_config(config)

    @classmethod
    def instantiate_model(cls, **kwargs: Any) -> Any:
        """
        Instantiate a model.
        """
        if cls.is_transformers_model():
            return cls.instantiate_transformers_model(**kwargs)
        elif cls.is_diffusers_model():
            return cls.instantiate_diffusers_model(**kwargs)
        else:
            model_class = cls.get_model_class()
            if model_class is not None:
                config = cls.merge_default_config(kwargs)
                assert isinstance(config, dict), "Configuration must be a dictionary."
                config_class = cls.get_config_class()
                if config_class is not None:
                    if cls.spread_config:
                        return model_class(config_class(**config))
                    return model_class(config_class(config))
                else:
                    if cls.spread_config:
                        return model_class(**config)
                    return model_class(config)
        import torch
        return torch.nn.Module()

    @classmethod
    def instantiate_and_load(
        cls,
        model_file: Optional[Union[str, List[str]]]=None,
        init_files: Optional[Dict[str, Union[str, List[str]]]]=None,
        init_weights: bool=False,
        device: Optional[Union[str, torch.device, Sequence[Union[str, torch.device]]]]=None,
        dtype: Optional[Union[str, torch.dtype]]=None,
        strict: bool=True,
        **kwargs: Any
    ) -> Any:
        """
        Instantiate a pretrained model.
        """
        import torch
        from .dtype_util import get_torch_dtype
        from .state_dict_util import load_state_dict
        from .init_util import no_init_weights
        from ..test_util import log_duration
        from ..log_util import logger

        is_cpu = device is None or (isinstance(device, str) and device == "cpu") or (isinstance(device, torch.device) and device.type == torch.device("cpu").type)
        if cls.dtype is not None:
            dtype = cls.dtype

        if cls.use_torch_jit:
            # TorchScript instantiation
            assert model_file is not None, "Model file is required for TorchScript instantiation."
            assert not bool(init_files), "Init files are not supported for TorchScript instantiation."
            with log_duration(f"Instantiating from TorchScript file {model_file}"):
                model = torch.jit.load(model_file, map_location="cpu" if is_cpu else device) # type: ignore[no-untyped-call]

            if cls.use_eval:
                model.eval()

            if cls.move_after_load and not is_cpu:
                with log_duration(f"Moving {type(model).__name__} to {device}"):
                    model.to(device, dtype=dtype)

            return model

        # Normal torch module instantiation, either diffusers, transformers or native torch

        if cls.is_transformers_model():
            from transformers.quantizers import AutoHfQuantizer as TransformersAutoQuantizer # type: ignore[import-not-found,import-untyped,unused-ignore]
            quant_cls = TransformersAutoQuantizer
        elif cls.is_diffusers_model():
            from diffusers.quantizers import DiffusersAutoQuantizer # type: ignore[attr-defined,import-not-found,unused-ignore]
            quant_cls = DiffusersAutoQuantizer

        instantiate_context = no_init_weights() if cls.no_init_weights and not init_weights else nullcontext()
        strict = cls.use_strict and strict
        instantiate_kwargs = kwargs

        if init_files is not None:
            assert cls.init_file_urls is not None, "Init file URLs are not defined."
            for init_file_name, init_file_path in init_files.items():
                if init_file_name in cls.init_file_urls:
                    instantiate_kwargs[init_file_name] = init_file_path

        with instantiate_context:
            with log_duration(f"Instantiating {cls.__name__}"):
                model = cls.instantiate_model(**instantiate_kwargs)

        if model_file is not None:
            load_target = cls.get_load_target(model)
            if dtype is not None and isinstance(dtype, str):
                dtype = get_torch_dtype(dtype)

            quantization_config = cls.get_quantization_config()
            if quantization_config is not None:
                logger.debug(f"Using quantization config {quantization_config}, {cls.pre_quantized=}")

                if isinstance(quantization_config, dict):
                    method = quantization_config.get("method", None)
                    assert method == "quanto", f"Invalid quantization method {method}."
                    quantizer = OptimumQuantoQuantizer(
                        weights=quantization_config.get("qtype", None),
                        pre_quantized=cls.pre_quantized
                    )
                else:
                    quantizer = quant_cls.from_config(quantization_config, pre_quantized=cls.pre_quantized)
                    use_keep_in_fp32_modules = (load_target._keep_in_fp32_modules is not None) and (
                        (dtype is not torch.float32) or hasattr(quantizer, "use_keep_in_fp32_modules")
                    )
            else:
                quantizer = None

            if quantizer is not None and not isinstance(quantizer, OptimumQuantoQuantizer):
                dtype = quantizer.update_torch_dtype(dtype)
                if use_keep_in_fp32_modules:
                    keep_in_fp32_modules = load_target._keep_in_fp32_modules
                else:
                    keep_in_fp32_modules = []

                for param in load_target.parameters():
                    # Disable gradient computation for all parameters
                    param.requires_grad = False

                logger.debug(f"Preprocessing model {type(load_target).__name__} with quantizer {type(quantizer).__name__}")
                quantizer.preprocess_model(
                    model=load_target,
                    device_map="auto",
                    keep_in_fp32_modules=keep_in_fp32_modules,
                )
            else:
                keep_in_fp32_modules = None

            logger.debug(f"Loading model from {model_file} into {type(load_target).__name__} with device {device} and dtype {dtype}")

            if isinstance(quantizer, OptimumQuantoQuantizer) and quantizer.pre_quantized:
                # We quantize and load weights in one step
                quantizer.requantize(
                    model=load_target,
                    checkpoint=model_file,
                    device=device, # type: ignore[arg-type]
                )
            elif cls.use_accelerate and cls.is_accelerate_available():
                # Load weights using accelerate
                from accelerate import dispatch_model

                for i, load_file in enumerate([model_file] if isinstance(model_file, str) else model_file):
                    device_map = None

                    if device is not None and cls.load_on_device and not is_cpu:
                        device_map = {}

                        if isinstance(device, list):
                            load_device = device[i]
                        else:
                            load_device = device

                        if isinstance(device, str):
                            device_parts = device.split(":")
                            if len(device_parts) == 2:
                                device_index = int(device_parts[1])
                            else:
                                device_index = 0
                        elif isinstance(device, torch.device):
                            device_index = device.index
                        else:
                            raise ValueError(f"Invalid device {device}.")

                        device_map[''] = device_index

                    if quantizer is not None and not isinstance(quantizer, OptimumQuantoQuantizer):
                        torch_dtype = quantizer.adjust_target_dtype(dtype)
                        device_map = quantizer.update_device_map(device_map)
                    else:
                        torch_dtype = dtype

                    with log_duration(f"Loading {load_file} into {type(load_target).__name__} using accelerate"):
                        unexpected = cls.load_checkpoint_into_model(
                            load_target,
                            load_file,
                            quantizer=quantizer,
                            device=device_map[''] if device_map is not None else None, # type: ignore[arg-type]
                            dtype=dtype,
                        )
                        if strict and unexpected:
                            raise ValueError(f"Unexpected keys in state dict: {unexpected}")

                if cls.load_on_device and cls.move_after_load and not is_cpu:
                    try:
                        dispatch_model(load_target, device_map)
                    except ValueError:
                        # Check for tensors that are still on the meta device.
                        # If these aren't replaced with a tensor on the correct
                        # device, dispatch_model will fail
                        for name, param in load_target.named_parameters():
                            if param.device.type == "meta":
                                logger.error(f"Parameter {name} is still on the meta device.")
                        raise
            else:
                for i, load_file in enumerate([model_file] if isinstance(model_file, str) else model_file):
                    with log_duration(f"Loading {load_file} into {type(load_target).__name__}"):
                        load_target.load_state_dict(
                            load_state_dict(
                                load_file,
                                device=device, # type: ignore[arg-type]
                                dtype=dtype,
                            ),
                            strict=strict and not isinstance(model_file, list)
                        )

            if hasattr(load_target, "tie_weights") and cls.tie_weights:
                load_target.tie_weights()

            if quantizer is not None:
                logger.debug(f"Postprocessing model {type(load_target).__name__} with quantizer {type(quantizer).__name__}")
                quantizer.postprocess_model(load_target)
                load_target.quantizer = quantizer # type: ignore[assignment]

            if cls.use_eval:
                load_target.eval()

            if device is not None and not isinstance(device, list) and quantizer is None and cls.load_on_device and cls.move_after_load and not is_cpu:
                with log_duration(f"Moving {type(load_target).__name__} to {device}"):
                    load_target.to(device, dtype=dtype)

            if cls.use_compile:
                logger.debug(f"Enabling graph compilation for {type(load_target).__name__}")
                with log_duration(f"Compiling {type(load_target).__name__}"):
                    load_target.compile() # type: ignore[no-untyped-call]

        cls.post_load_hook(model)
        return model

    @classmethod
    def instantiate_and_load_from_url(
        cls,
        model_file: Optional[Union[str, List[str]]]=None,
        init_files: Optional[Dict[str, Union[str, List[str]]]]=None,
        init_weights: bool=False,
        device: Optional[Union[str, torch.device, Sequence[Union[str, torch.device]]]]=None,
        dtype: Optional[Union[str, torch.dtype]]=None,
        strict: bool=True,
        check_size: bool=False,
        **kwargs: Any
    ) -> Any:
        """
        Instantiate a pretrained model from the model URL to a file
        """
        files_to_download: List[Tuple[str, str]] = []
        if cls.model_url is None:
            assert model_file is None, f"This pretrained class definition does not have a model URL, nothing to download to {model_file}."
        else:
            assert model_file is not None, "Local path to download the model file(s) is/are required."
            if isinstance(cls.model_url, str):
                assert isinstance(model_file, str), f"Model URL is a string, but model file is not: {model_file}"
                files_to_download.append((cls.model_url, model_file))
            else:
                assert isinstance(model_file, list), f"Model URL is a list, but model file is not: {model_file}"
                assert len(cls.model_url) == len(model_file), f"Model URL length {len(cls.model_url)} does not match model file length {len(model_file)}."
                for model_url, model_path in zip(cls.model_url, model_file):
                    files_to_download.append((model_url, model_path))

        if cls.init_file_urls is None:
            assert init_files is None, f"This pretrained class definition does not have init file URLs, nothing to download to {init_files}."
        else:
            assert init_files is not None, "Local paths to download the init files are required."
            init_file_names = set(init_files.keys())
            defined_file_names = set(cls.init_file_urls.keys())
            assert init_file_names == defined_file_names, f"Init file names {init_file_names} do not match defined init file names {defined_file_names}."
            for init_file_name, init_file_path in init_files.items():
                if isinstance(cls.init_file_urls[init_file_name], str):
                    assert isinstance(init_file_path, str), f"Init file URL is a string, but init file path is not: {init_file_path}"
                    files_to_download.append((cls.init_file_urls[init_file_name], init_file_path)) # type: ignore[arg-type]
                else:
                    assert isinstance(init_file_path, list), f"Init file URL is a list, but init file path is not: {init_file_path}"
                    assert len(cls.init_file_urls[init_file_name]) == len(init_file_path), f"Init file URL length {len(cls.init_file_urls[init_file_name])} does not match init file path length {len(init_file_path)}."
                    for init_file_url, init_file_path in zip(cls.init_file_urls[init_file_name], init_file_path):
                        files_to_download.append((init_file_url, init_file_path))

        from ..download_util import check_download_files
        check_download_files(
            files_to_download,
            check_size=check_size
        )

        return cls.instantiate_and_load(
            model_file=model_file,
            init_files=init_files,
            init_weights=init_weights,
            device=device,
            dtype=dtype,
            strict=strict,
            **kwargs
        )

    @classmethod
    def instantiate_and_load_from_url_to_dir(
        cls,
        model_dir: str,
        init_weights: bool=False,
        device: Optional[Union[str, torch.device, Sequence[Union[str, torch.device]]]]=None,
        dtype: Optional[Union[str, torch.dtype]]=None,
        strict: bool=True,
        check_size: bool=False,
        **kwargs: Any
    ) -> Any:
        """
        Instantiate a pretrained model from the model URL to a directory
        """
        from ..download_util import get_file_name_from_url
        model_file: Optional[Union[str, List[str]]] = None
        if cls.model_url is not None:
            if isinstance(cls.model_url, str):
                model_file = os.path.join(model_dir, get_file_name_from_url(cls.model_url))
            else:
                model_file = [os.path.join(model_dir, get_file_name_from_url(url)) for url in cls.model_url]
        init_files: Optional[Dict[str, Union[str, List[str]]]] = None
        if cls.init_file_urls is not None:
            init_files = {}
            for init_file_name, init_file_url in cls.init_file_urls.items():
                if isinstance(init_file_url, str):
                    init_files[init_file_name] = os.path.join(model_dir, get_file_name_from_url(init_file_url))
                else:
                    init_files[init_file_name] = [os.path.join(model_dir, get_file_name_from_url(url)) for url in init_file_url]
        else:
            init_files = None

        return cls.instantiate_and_load_from_url(
            model_file=model_file,
            init_files=init_files,
            init_weights=init_weights,
            device=device,
            dtype=dtype,
            strict=strict,
            check_size=check_size,
            **kwargs
        )

    @classmethod
    def load_checkpoint_into_model(
        cls,
        model: torch.nn.Module,
        model_file: str,
        device: Optional[torch.device]=None,
        dtype: Optional[torch.dtype]=None,
        quantizer: Optional[Union[TransformersAutoQuantizer, DiffusersAutoQuantizer]]=None,
        keep_in_fp32_modules: Optional[List[str]]=None,
    ) -> List[str]:
        """
        Loads a checkpoint into a model, returning unexpected keys.

        :see https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/model_loading_utils.py#L170:
        """
        import torch
        from taproot.util import load_state_dict, logger, empty_cache
        from accelerate.utils import set_module_tensor_to_device # type: ignore[import-not-found,import-untyped,unused-ignore]
        if quantizer is None:
            device = device or torch.device("cpu")

        dtype = dtype or torch.float32
        is_quantized = quantizer is not None
        is_quant_method_bnb = getattr(model, "quantization_method", None) == "bitsandbytes"

        use_check_quantized_param = "check_quantized_param" in dir(quantizer)
        use_check_if_quantized_param = "check_if_quantized_param" in dir(quantizer)

        empty_state_dict = model.state_dict()
        state_dict = load_state_dict(model_file)
        unexpected_keys = [param_name for param_name in state_dict if param_name not in empty_state_dict]

        def is_quantized_param(param: torch.Tensor) -> bool:
            if not is_quantized:
                return False
            if use_check_quantized_param:
                return bool(quantizer.check_quantized_param(model, param, param_name, state_dict, param_device=device)) # type: ignore[union-attr]
            elif use_check_if_quantized_param:
                return bool(quantizer.check_if_quantized_param(model, param, param_name, state_dict, param_device=device)) # type: ignore[union-attr]
            return False

        for param_name, param in state_dict.items():
            if param_name not in empty_state_dict:
                continue

            if not torch.is_tensor(param): # type: ignore[no-untyped-call]
                continue

            set_module_kwargs = {}
            # We convert floating dtypes to the `dtype` passed. We also want to keep the buffers/params
            # in int/uint/bool and not cast them.
            # TODO: revisit cases when param.dtype == torch.float8_e4m3fn
            if torch.is_floating_point(param): # type: ignore[arg-type]
                if (
                    keep_in_fp32_modules is not None
                    and any(
                        module_to_keep_in_fp32 in param_name.split(".") for module_to_keep_in_fp32 in keep_in_fp32_modules
                    )
                    and dtype == torch.float16
                ):
                    param = param.to(torch.float32) # type: ignore[union-attr]
                    set_module_kwargs["dtype"] = torch.float32
                else:
                    param = param.to(dtype) # type: ignore[union-attr]
                    set_module_kwargs["dtype"] = dtype

            # bnb params are flattened.
            if empty_state_dict[param_name].shape != param.shape: # type: ignore[union-attr]
                if (
                    is_quantized
                    and is_quant_method_bnb
                    and getattr(quantizer, "pre_quantized", False)
                    and "check_quantized_param_shape" in dir(quantizer)
                    and is_quantized_param(param) # type: ignore[arg-type]
                ):
                    quantizer.check_quantized_param_shape(param_name, empty_state_dict[param_name], param) # type: ignore[union-attr]
                elif not is_quant_method_bnb:
                    raise ValueError(
                        f"Cannot load because {param_name} expected shape {empty_state_dict[param_name]}, but got {param.shape}. If you want to instead overwrite randomly initialized weights, please make sure to pass both `low_cpu_mem_usage=False` and `ignore_mismatched_sizes=True`. For more information, see also: https://github.com/huggingface/diffusers/issues/1619#issuecomment-1345604389 as an example." # type: ignore[union-attr]
                    )

            if (
                is_quantized
                and is_quantized_param(param) # type: ignore[arg-type]
            ):
                quantizer.create_quantized_param(model, param, param_name, device, state_dict, unexpected_keys) # type: ignore[union-attr]
            else:
                set_module_tensor_to_device(model, param_name, "cpu" if device is None else device, value=param, **set_module_kwargs)

        empty_cache()
        return unexpected_keys
