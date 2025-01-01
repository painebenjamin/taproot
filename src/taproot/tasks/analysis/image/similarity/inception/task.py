from __future__ import annotations

from typing import Optional, Dict, List, Tuple, TYPE_CHECKING

from taproot.constants import *
from taproot.util import to_bchw_tensor
from taproot.tasks.base import Task

from .pretrained import PretrainedInception3

if TYPE_CHECKING:
    from torch import Tensor
    from taproot.hinting import ImageType
    from torchvision.models import Inception3 # type: ignore[import-not-found,import-untyped,unused-ignore]

__all__ = ["InceptionImageSimilarity"]

class InceptionImageSimilarity(Task):
    """
    Calculate the euclidean distance between the feature
    representations of two images using the Inception3 model.
    """

    """Global Task Metadata"""
    task = "image-similarity"
    model = "inception-v3"
    default = False
    display_name = "Inception Image Similarity (FID)"
    use_gpu = True
    static_memory_gb =  79.03 / 1024.0 # 79 MB, measured
    static_gpu_memory_gb = 51.49 / 1024.0 # 51 MB, measured

    """Authorship metadata"""
    author = "Christian Szegedy"
    author_url = "https://arxiv.org/abs/1512.00567"
    author_additional = ["Vincent Vanhoucke", "Sergey Ioffe", "Jonathon Shlens", "Zbigniew Wojna"]
    author_affiliations = ["Google Research", "University College London"]
    author_journal = "CoRR"
    author_journal_title = "Rethinking the Inception Architecture for Computer Vision"
    author_journal_volume = "1512.00567"
    author_journal_year = 2015

    """License metadata"""
    license = LICENSE_APACHE

    """Internal Task Properties"""
    normalized_means = [0.485, 0.456, 0.406]
    normalized_stds = [0.229, 0.224, 0.225]
    inception_model: Inception3

    @classmethod
    def required_files(cls, allow_optional: bool=True) -> List[str]:
        """
        The files required to run this task.
        """
        return PretrainedInception3.get_required_files()

    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        The packages required to run this task.
        """
        return {
            "torch": TORCH_VERSION_SPEC,
            "safetensors": SAFETENSORS_VERSION_SPEC,
            "torchvision": TORCHVISION_VERSION_SPEC,
            "scipy": SCIPY_VERSION_SPEC,
            "numpy": NUMPY_VERSION_SPEC,
            "pil": PILLOW_VERSION_SPEC
        }

    """Internal methods"""

    def get_inception_features(self, image_tensor: Tensor) -> Tensor:
        """
        Get the inception features for an image.
        """
        import torch
        with torch.inference_mode():
            return self.inception_model(image_tensor) # type: ignore[no-any-return]

    def get_statistics(
        self,
        features: Tensor,
        epsilon: float = 1e-6
    ) -> Tuple[Tensor, Tensor]:
        """
        Get the mean and covariance of the features.
        """
        import torch
        mu = features.mean(dim=0)
        features_centered = features - mu
        sigma = torch.mm(features_centered.t(), features_centered) / (features.shape[0] - 1)
        sigma = sigma + epsilon * torch.eye(sigma.shape[0], device=sigma.device) # Regularize
        return mu, sigma

    """Overrides"""

    def load(self) -> None:
        """
        Load the model and any other resources required for the task.
        """
        self.inception_model = PretrainedInception3.instantiate_and_load_from_url_to_dir(
            self.model_dir,
            device=self.device,
            dtype=self.dtype
        )

    def __call__( # type: ignore[override]
        self,
        *,
        left: ImageType,
        right: ImageType,
        fid_epsilon: float = 1e-6
    ) -> float:
        """
        Calculate the euclidean distance between two image representations.
        Normalized to the maximum distance possible (sqrt of feature size).

        :param left: The left image.
        :param right: The right image.
        :param fid_epsilon: The epsilon value for FID calculation.
        :return: The euclidean distance between the two images.
        """
        import torch
        left_tensor = to_bchw_tensor(left, mean=self.normalized_means, std=self.normalized_stds, resize=(299, 299))
        right_tensor = to_bchw_tensor(right, mean=self.normalized_means, std=self.normalized_stds, resize=(299, 299))
        num_images = min(left_tensor.shape[0], right_tensor.shape[0])

        if num_images == 1:
            # Euclidean distance between two images
            features_l, features_r = self.get_inception_features(
                torch.cat([left_tensor, right_tensor], dim=0).to(self.device, dtype=self.dtype)
            )
            distance = torch.norm(features_l - features_r).item()
            max_distance = features_l.shape[-1] ** 0.5
            return float(distance / max_distance)
        else:
            # FID calculation between two sets of images
            # First exec ~= 300ms, second 15ms for 1 image, 300ms for 2048 images
            features_l = self.get_inception_features(
                left_tensor.to(self.device, dtype=self.dtype)
            )
            features_r = self.get_inception_features(
                right_tensor.to(self.device, dtype=self.dtype)
            )
            # Get statistics
            mu_l, sigma_l = self.get_statistics(features_l, epsilon=fid_epsilon)
            mu_r, sigma_r = self.get_statistics(features_r, epsilon=fid_epsilon)
            # Calulate matrix sqrt - can be slow for large matrices
            sigma = torch.mm(sigma_l, sigma_r)
            u, s, v = torch.svd(sigma)
            covsqrt = torch.mm(u, torch.mm(s.diag().sqrt(), v.t()))
            # Calculate FID
            diff = mu_l - mu_r
            diff = diff.dot(diff)
            trace_l = torch.trace(sigma_l)
            trace_r = torch.trace(sigma_r)
            trace_sqrt = 2 * torch.trace(covsqrt)
            fid = diff + trace_l + trace_r - trace_sqrt
            return float(fid.item())
