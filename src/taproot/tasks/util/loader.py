from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Type, TYPE_CHECKING
from taproot.util import (
    PretrainedModelMixin,
    logger,
    get_torch_dtype,
    empty_cache,
    maybe_use_tqdm
)

if TYPE_CHECKING:
    import torch
    from taproot.tasks.base import Task

__all__ = ["Loader", "PretrainedLoader", "TaskLoader"]

class Loader:
    """
    A class that aids in loading, using and unloading pretrained models.
    """
    loaded: Dict[str, Any]
    device: Optional[torch.device]
    dtype: Optional[torch.dtype]

    def __init__(
        self,
        directory: str,
        device: Optional[Union[str, torch.device]]=None,
        dtype: Optional[Union[str, torch.dtype]]=None,
    ) -> None:
        self.directory = directory
        self.loaded = {}
        if device is not None and isinstance(device, str):
            import torch
            self.device = torch.device(device)
        else:
            self.device = device
        if dtype is not None and isinstance(dtype, str):
            self.dtype = get_torch_dtype(dtype)
        else:
            self.dtype = dtype

    def get_required_files(self) -> List[str]:
        """
        Gets the required files for all models.
        """
        raise NotImplementedError()

    def get_loadable_names(self) -> List[str]:
        """
        Gets the names of all models that can be loaded.
        """
        raise NotImplementedError()

    def load_by_name(self, model_name: str, offload: bool=False) -> None:
        """
        Load a model by name.
        """
        raise NotImplementedError()

    def unload_by_name(self, model_name: str) -> None:
        """
        Unload a model by name.
        """
        raise NotImplementedError()

    def offload_by_name(self, model_name: str) -> None:
        """
        Offload a model by name (from GPU to CPU).
        """
        raise NotImplementedError()

    def onload_by_name(self, model_name: str) -> None:
        """
        Onload a model by name (from CPU to GPU).
        """
        raise NotImplementedError()

    def compile_by_name(self, model_name: str) -> None:
        """
        Compile a model by name.
        """
        raise NotImplementedError()

    def load(
        self,
        loadable_names: Optional[List[str]] = None,
        offload_names: Optional[List[str]] = None,
        use_tqdm: bool=False
    ) -> None:
        """
        Load all models.
        """
        if loadable_names is None:
            loadable_names = self.get_loadable_names()

        offloaded_names = []
        for model_name in maybe_use_tqdm(
            loadable_names,
            use_tqdm=use_tqdm and bool(loadable_names),
        ):
            self.load_by_name(
                model_name,
                offload=offload_names is not None and model_name in offload_names
            )
            if offload_names is not None and model_name in offload_names:
                logger.debug(f"Offloading {model_name} after loading")
                self.offload_by_name(model_name)
                offloaded_names.append(model_name)

        if offload_names is not None:
            still_to_offload = set(offload_names) - set(offloaded_names)
            for model_name in still_to_offload:
                self.offload_by_name(model_name)

    def unload(self) -> None:
        """
        Unload all models.
        """
        for model_name in list(self.get_loadable_names()):
            self.unload_by_name(model_name)
        empty_cache()

    def offload(self) -> None:
        """
        Offload all models (from GPU to CPU).
        """
        for model_name in self.get_loadable_names():
            self.offload_by_name(model_name)

    def onload(self) -> None:
        """
        Onload all models (from CPU to GPU).
        """
        for model_name in self.get_loadable_names():
            self.onload_by_name(model_name)

    def compile(self, use_tqdm: bool=False) -> None:
        """
        Compile all models.
        """
        loadable_names = self.get_loadable_names()
        for model_name in maybe_use_tqdm(
            loadable_names,
            use_tqdm=use_tqdm and bool(loadable_names),
        ):
            self.compile_by_name(model_name)

    def __getattr__(self, model_name: str) -> Any:
        """
        Get a loaded model
        """
        try:
            return self.loaded[model_name]
        except KeyError:
            try:
                self.load_by_name(model_name)
                return getattr(self, model_name)
            except KeyError:
                raise AttributeError(f"Model/Task {model_name} not found")

class PretrainedLoader(Loader):
    """
    A class that aids in loading, using and unloading pretrained models.
    """
    loaded: Dict[str, Any]
    device: Optional[torch.device]
    dtype: Optional[torch.dtype]

    def __init__(
        self,
        directory: str,
        device: Optional[Union[str, torch.device]]=None,
        dtype: Optional[Union[str, torch.dtype]]=None,
        **models: Type[PretrainedModelMixin]
    ) -> None:
        super(PretrainedLoader, self).__init__(directory, device, dtype)
        self.models = models

    def get_required_files(self) -> List[str]:
        """
        Gets the required files for all models.
        """
        file_list: List[str] = []
        for model in self.models.values():
            file_list.extend(model.get_required_files())
        return file_list

    def get_loadable_names(self) -> List[str]:
        """
        Gets the names of all models that can be loaded.
        """
        return list(self.models.keys())

    def load_by_name(self, name: str, offload: bool=False) -> None:
        """
        Loads a model by name.
        """
        if name in self.loaded:
            return self.onload_by_name(name)
        model_class = self.models.get(name, None)
        if model_class is None:
            raise KeyError(f"Model {name} not found")
        if offload and model_class.quantization is None:
            device = None
        else:
            device = self.device
        self.loaded[name] = model_class.instantiate_and_load_from_url_to_dir(
            self.directory,
            device=device,
            dtype=self.dtype
        )

    def compile_by_name(self, name: str) -> None:
        """
        Compile a model by name.
        """
        if name not in self.loaded:
            self.load_by_name(name) # Will raise an error if not found
        self.loaded[name].compile() # TODO: load/save cached

    def unload_by_name(self, name: str) -> None:
        """
        Unload all models.
        """
        if name in self.loaded:
            del self.loaded[name]

    def offload_by_name(self, name: str) -> None:
        """
        Offload all models (from GPU to CPU).
        """
        if name in self.loaded:
            self.loaded[name].to("cpu")

    def onload_by_name(self, name: str) -> None:
        """
        Onload all models (from CPU to GPU).
        """
        if name in self.loaded:
            self.loaded[name].to(self.device)
        else:
            self.load_by_name(name)

class TaskLoader(Loader):
    """
    A version of the pretrained loader for use with component tasks.
    """
    def __init__(
        self,
        directory: str,
        device: Optional[Union[str, torch.device]]=None,
        dtype: Optional[Union[str, torch.dtype]]=None,
        **tasks: Type[Task]
    ) -> None:
        super(TaskLoader, self).__init__(directory, device, dtype)
        self.tasks = tasks

    def get_required_files(self) -> List[str]:
        """
        Gets the required files for all tasks.
        """
        file_list: List[str] = []
        for task_class in self.tasks.values():
            file_list.extend(task_class.required_files())
        return file_list

    def get_loadable_names(self) -> List[str]:
        """
        Gets the names of all tasks that can be loaded.
        """
        return list(self.tasks.keys())

    def load_by_name(self, task_name: str, offload: bool=False) -> None:
        """
        Load a task by name.
        """
        task_class = self.tasks.get(task_name, None)
        if task_class is None:
            raise KeyError(f"Task {task_name} not found")
        device = None if offload else self.device
        task = task_class({
            "gpu_index": 0 if not device else device.index,
            "dtype": str(self.dtype),
            "model_dir": self.directory
        })
        task.load()
        self.loaded[task_name] = task

    def compile_by_name(self, task_name: str) -> None:
        """
        Compile a task.
        """
        if task_name not in self.loaded:
            self.load_by_name(task_name)
        self.loaded[task_name].compile()

    def unload_by_name(self, task_name: str) -> None:
        """
        Unload a task.
        """
        if task_name in self.loaded:
            self.loaded[task_name].unload()
            del self.loaded[task_name]

    def offload_by_name(self, task_name: str) -> None:
        """
        Offload a task (from GPU to CPU).
        """
        if task_name in self.loaded:
            self.loaded[task_name].offload()

    def onload_by_name(self, task_name: str) -> None:
        """
        Onload a task (from CPU to GPU).
        """
        if task_name in self.loaded:
            self.loaded[task_name].onload()
        else:
            self.load_by_name(task_name)
