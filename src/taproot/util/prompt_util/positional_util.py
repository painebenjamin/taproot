from __future__ import annotations

from typing import Optional, Tuple, List, Union, TYPE_CHECKING

from taproot.constants import *
from ..image_util import to_bchw_tensor

if TYPE_CHECKING:
    import torch
    from ...hinting import ImageType

__all__ = [
    "SpatioTemporalPrompt",
    "EncodedPrompt",
    "EncodedPrompts"
]

class SpatioTemporalPrompt:
    """
    A class to store a prompt with optional position and time information.
    """
    mask: Optional[torch.Tensor]

    def __init__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        weight: float = 1.0,
        position: Optional[Tuple[int, int, int, int]] = None,
        time: Optional[Tuple[int, int]] = None,
        mask: Optional[ImageType] = None,
    ) -> None:
        """
        :param prompt: The prompt to store. Required.
        :param negative_prompt: The negative prompt to store. Defaults to None.
        :param position: The position of the prompt in the format (left, top, right, bottom). Defaults to None.
        :param time: The time of the prompt in the format (start_frame, end_frame). Defaults to None.
        """
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.weight = weight
        self.position = position
        self.time = time

        if mask is not None:
            self.mask = to_bchw_tensor(mask, num_channels=1)[0]
        else:
            self.mask = None

class EncodedPrompt:
    """
    A class to store the embeddings of a prompt, with optional position and time information.
    """
    def __init__(
        self,
        embeddings: torch.Tensor,
        weight: float = 1.0,
        mask: Optional[torch.Tensor] = None,
        position: Optional[Tuple[int, int, int, int]] = None,
        time: Optional[Tuple[int, int]] = None,
        negative_embeddings: Optional[torch.Tensor] = None,
        pooled_embeddings: Optional[torch.Tensor] = None,
        negative_pooled_embeddings: Optional[torch.Tensor] = None,
        space: IMAGE_SPACE_LITERAL = "pixel",
        mask_space: IMAGE_SPACE_LITERAL = "pixel",
    ) -> None:
        """
        :param embeddings: The embeddings of the prompt. Required.
        :param weight: The weight of the prompt. Defaults to 1.0.
        :param position: The position of the prompt in the format (left, top, right, bottom). Defaults to None.
        :param time: The time of the prompt in the format (start_frame, end_frame). Defaults to None.
        :param negative_embeddings: The negative embeddings of the prompt. Defaults to None.
        :param pooled_embeddings: The pooled embeddings of the prompt. Defaults to None.
        :param negative_pooled_embeddings: The negative pooled embeddings of the prompt. Defaults to None.
        :param space: The space of the embeddings. Defaults to "pixel".
        """
        self.embeddings = embeddings
        self.weight = weight
        self.mask = mask
        self.mask_space = mask_space
        self.position = position
        self.time = time
        self.negative_embeddings = negative_embeddings
        self.pooled_embeddings = pooled_embeddings
        self.negative_pooled_embeddings = negative_pooled_embeddings
        self.space = space

    def get_mask(
        self,
        position: Optional[Tuple[int, int, int, int]] = None,
        space: IMAGE_SPACE_LITERAL = "latent",
        latent_scale_factor: int = 8,
    ) -> Optional[torch.Tensor]:
        """
        Get the mask of the prompt.

        :return: The mask of the prompt
        """
        import torch

        if self.mask is None:
            return None

        if space == "latent" and self.mask_space == "pixel":
            mask = torch.nn.functional.interpolate(
                self.mask.unsqueeze(0),
                scale_factor=1 / latent_scale_factor,
                mode="nearest"
            ).squeeze(0)
        elif space == "pixel" and self.mask_space == "latent":
            mask = torch.nn.functional.interpolate(
                self.mask.unsqueeze(0),
                scale_factor=latent_scale_factor,
                mode="nearest"
            ).squeeze(0)
        else:
            mask = self.mask

        if position is not None:
            # Crop the mask to the position
            left, top, right, bottom = position
            mask = mask[:, top:bottom, left:right]

        return mask # type: ignore[no-any-return]

    def get_position_overlap_ratio(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
        space: IMAGE_SPACE_LITERAL = "latent",
        latent_scale_factor: int = 8,
    ) -> float:
        """
        Calculate the ratio of the overlap between the position and the given region.

        :param left: The left coordinate of the region.
        :param right: The right coordinate of the region.
        :param top: The top coordinate of the region.
        :param bottom: The bottom coordinate of the region.
        :param space: The space of the region. Defaults to "latent".
        :return: The ratio of the overlap between the position and the given region
        """
        if self.position is None:
            return 1.0

        position_left, position_top, position_right, position_bottom = self.position

        if space == "latent" and self.space == "pixel":
            position_left /= latent_scale_factor # type: ignore[assignment]
            position_right /= latent_scale_factor # type: ignore[assignment]
            position_top /= latent_scale_factor # type: ignore[assignment]
            position_bottom /= latent_scale_factor # type: ignore[assignment]
        elif space == "pixel" and self.space == "latent":
            position_left *= latent_scale_factor
            position_right *= latent_scale_factor
            position_top *= latent_scale_factor
            position_bottom *= latent_scale_factor

        overlap_left = max(left, position_left)
        overlap_right = min(right, position_right)
        overlap_top = max(top, position_top)
        overlap_bottom = min(bottom, position_bottom)

        overlap_area = max(0, overlap_right - overlap_left) * max(0, overlap_bottom - overlap_top)
        position_area = (right - left) * (bottom - top)

        return overlap_area / position_area

    def get_time_overlap_ratio(
        self,
        start: int,
        end: int,
    ) -> float:
        """
        Calculate the ratio of the overlap between the time and the given region.

        :param start: The start frame of the region.
        :param end: The end frame of the region.
        """
        if self.time is None:
            return 1.0
        time_start, time_end = self.time
        overlap_start = max(start, time_start)
        overlap_end = min(end, time_end)
        overlap_duration = max(0, overlap_end - overlap_start)
        time_duration = time_end - time_start
        return overlap_duration / time_duration

class EncodedPrompts:
    """
    A class to store multiple EncodedPrompt objects.
    """
    def __init__(
        self,
        do_classifier_free_guidance: bool=True,
        do_perturbed_attention_guidance: bool=False,
        *prompts: EncodedPrompt
    ) -> None:
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.do_perturbed_attention_guidance = do_perturbed_attention_guidance
        self.prompts = list(prompts)

    def scale_prompt_masks(self, height: int, width: int) -> None:
        """
        Scale the masks of the prompts by the given scale factor.

        :param scale_factor: The scale factor to apply.
        """
        import torch
        for prompt in self.prompts:
            if prompt.mask is not None:
                prompt.mask = torch.nn.functional.interpolate(
                    prompt.mask.unsqueeze(0),
                    size=(height, width),
                    mode="nearest"
                ).squeeze(0)

    def add_prompt(self, prompt: EncodedPrompt) -> None:
        """
        Add a prompt to the list of prompts.

        :param prompt: The prompt to add.
        """
        self.prompts.append(prompt)

    def get_embeddings(
        self,
        position: Optional[Tuple[int, int, int, int]] = None,
        time: Optional[Tuple[int, int]] = None,
        latent_scale_factor: int = 8,
        space: IMAGE_SPACE_LITERAL = "latent",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Get the embeddings of the prompts that apply to the given position and time.

        When no position or time are specified, either in this method or on a constituent prompt object, they
        are treated as always applying and thus the embeddings are always factored into the final result.

        Only the positive embeddings are guaranteed, other embeddings depend on the prompts encoded in this object.

        :param position: The position of the embeddings in the format (left, right, top, bottom). Defaults to None.
        :param time: The time of the embeddings in the format (start_frame, end_frame). Defaults to None.
        :return: The embeddings, pooled embeddings, and mask of the prompts that apply to the given position and time.
        """
        import torch

        masks: List[Optional[torch.Tensor]] = []
        embeddings: List[Tuple[torch.Tensor, float]] = []
        negative_embeddings: List[Tuple[torch.Tensor, float]] = []
        pooled_embeddings: List[Tuple[torch.Tensor, float]] = []
        negative_pooled_embeddings: List[Tuple[torch.Tensor, float]] = []

        for i, prompt in enumerate(self.prompts):
            if position is not None:
                overlap_ratio = prompt.get_position_overlap_ratio(
                    *position,
                    space=space,
                    latent_scale_factor=latent_scale_factor
                )
                if overlap_ratio == 0.0:
                    continue
            else:
                overlap_ratio = 1.0

            if time is not None:
                time_overlap_ratio = prompt.get_time_overlap_ratio(*time)
                if time_overlap_ratio == 0.0:
                    continue
            else:
                time_overlap_ratio = 1.0

            weight = overlap_ratio * time_overlap_ratio * prompt.weight

            embeddings.append((prompt.embeddings, weight))
            masks.append(prompt.get_mask(position=position, space=space, latent_scale_factor=latent_scale_factor))
            if prompt.negative_embeddings is not None:
                negative_embeddings.append((prompt.negative_embeddings, weight))
            if prompt.pooled_embeddings is not None:
                pooled_embeddings.append((prompt.pooled_embeddings, weight))
            if prompt.negative_pooled_embeddings is not None:
                negative_pooled_embeddings.append((prompt.negative_pooled_embeddings, weight))

        if not embeddings:
            raise ValueError("No embeddings found for the given position; you should have at least one set of embeddings that always applies.")

        has_mask = any(mask is not None for mask in masks)
        total_embeddings_weight = sum(weight for _, weight in embeddings)

        if has_mask:
            non_empty_masks = [mask for mask in masks if mask is not None]
            mask_shape = non_empty_masks[0].shape
            mask_type = non_empty_masks[0].dtype
            assert all(mask.shape == mask_shape for mask in non_empty_masks[1:]), "All masks must have the same shape."
            masks_tensor = torch.stack([
                mask if mask is not None
                else torch.ones(mask_shape, dtype=mask_type)
                for mask in masks
            ])
        else:
            masks_tensor = None

        embeddings_tensor = torch.stack([
            embedding * weight
            for embedding, weight
            in embeddings
        ])
        if not has_mask:
            embeddings_tensor = embeddings_tensor.sum(dim=0) / total_embeddings_weight

        negative_embeddings_tensor: Optional[torch.Tensor] = None
        if negative_embeddings:
            total_negative_embeddings_weight = sum(weight for _, weight in negative_embeddings)
            negative_embeddings_tensor = torch.stack([
                embedding * weight
                for embedding, weight
                in negative_embeddings
            ])
            if not has_mask:
                negative_embeddings_tensor = negative_embeddings_tensor.sum(dim=0) / total_negative_embeddings_weight

        pooled_embeddings_tensor: Optional[torch.Tensor] = None
        if pooled_embeddings:
            total_pooled_embeddings_weight = sum(weight for _, weight in pooled_embeddings)
            pooled_embeddings_tensor = torch.stack([ 
                embedding * weight
                for embedding, weight
                in pooled_embeddings
            ])
            if not has_mask:
                pooled_embeddings_tensor = pooled_embeddings_tensor.sum(dim=0) / total_pooled_embeddings_weight

        negative_pooled_embeddings_tensor: Optional[torch.Tensor] = None
        if negative_pooled_embeddings:
            total_negative_pooled_embeddings_weight = sum(weight for _, weight in negative_pooled_embeddings)
            negative_pooled_embeddings_tensor = torch.stack([
                embedding * weight
                for embedding, weight
                in negative_pooled_embeddings
            ])
            if not has_mask:
                negative_pooled_embeddings_tensor = negative_pooled_embeddings_tensor.sum(dim=0) / total_negative_pooled_embeddings_weight

        stack_dim = 1 if has_mask else 0
        if self.do_classifier_free_guidance:
            if negative_embeddings_tensor is None:
                negative_embeddings_tensor = torch.zeros_like(embeddings_tensor)
            if pooled_embeddings_tensor is not None and negative_pooled_embeddings_tensor is None:
                negative_pooled_embeddings_tensor = torch.zeros_like(pooled_embeddings_tensor)

            if self.do_perturbed_attention_guidance:
                embeddings_tensor = torch.cat([
                    negative_embeddings_tensor,
                    embeddings_tensor,
                    embeddings_tensor
                ], dim=stack_dim)
                if pooled_embeddings_tensor is not None:
                    pooled_embeddings_tensor = torch.cat([
                        negative_pooled_embeddings_tensor, # type: ignore[list-item]
                        pooled_embeddings_tensor,
                        pooled_embeddings_tensor
                    ], dim=stack_dim)
            else:
                embeddings_tensor = torch.cat([
                    negative_embeddings_tensor,
                    embeddings_tensor
                ], dim=stack_dim)
                if pooled_embeddings_tensor is not None:
                    pooled_embeddings_tensor = torch.cat([
                        negative_pooled_embeddings_tensor, # type: ignore[list-item]
                        pooled_embeddings_tensor
                    ], dim=stack_dim)
        elif self.do_perturbed_attention_guidance:
            embeddings_tensor = torch.cat([
                embeddings_tensor,
                embeddings_tensor
            ], dim=stack_dim)
            if pooled_embeddings_tensor is not None:
                pooled_embeddings_tensor = torch.cat([
                    pooled_embeddings_tensor,
                    pooled_embeddings_tensor
                ], dim=stack_dim)

        if device is not None:
            embeddings_tensor = embeddings_tensor.to(device, dtype=dtype)
            if pooled_embeddings_tensor is not None:
                pooled_embeddings_tensor = pooled_embeddings_tensor.to(device, dtype=dtype)
            if masks_tensor is not None:
                masks_tensor = masks_tensor.to(device, dtype=dtype)
        elif dtype is not None:
            embeddings_tensor = embeddings_tensor.to(dtype=dtype)
            if pooled_embeddings_tensor is not None:
                pooled_embeddings_tensor = pooled_embeddings_tensor.to(dtype=dtype)
            if masks_tensor is not None:
                masks_tensor = masks_tensor.to(dtype=dtype)

        return embeddings_tensor, pooled_embeddings_tensor, masks_tensor
