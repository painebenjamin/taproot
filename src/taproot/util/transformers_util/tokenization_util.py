from __future__ import annotations

from typing import Dict, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AddedToken # type: ignore[import-untyped,import-not-found,unused-ignore]

__all__ = ["get_added_token_dict"]

def get_added_token_dict(
    param_dict: Dict[
        Union[str, int],
        Dict[str, Union[str, bool]]
    ]
) -> Dict[int, AddedToken]:
    """
    Convert a dictionary of dictionaries to a dictionary of AddedToken objects.
    """
    from transformers import AddedToken # type: ignore[import-untyped,import-not-found,unused-ignore]
    return dict([
        (int(key), AddedToken(**value))
        for key, value in param_dict.items()
    ])
