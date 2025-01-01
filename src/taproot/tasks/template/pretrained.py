from __future__ import annotations

from typing import Any, Dict, Optional, Type, List, Union, TYPE_CHECKING
from taproot.util import PretrainedModelMixin

if TYPE_CHECKING:
    from .model import TemplateModel

__all__ = ["PretrainedTemplateModel"]

class PretrainedTemplateModel(PretrainedModelMixin):
    """
    Pretrained Template Model.
    """
    model_url: Optional[Union[str, List[str]]] = "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/template-model.fp16.safetensors"
    init_file_urls: Optional[Dict[str, Union[str, List[str]]]] = None

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        Returns the default configuration for the model.
        """
        # return {"model_type": "gpt2", "num_labels": 2}
        return None

    @classmethod
    def get_model_class(cls) -> Type[TemplateModel]:
        """
        Returns the model class.
        """
        from .model import TemplateModel
        return TemplateModel
