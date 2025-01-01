from __future__ import annotations

from typing import Dict, List, Optional, Union, TYPE_CHECKING
from typing_extensions import Literal

from taproot.constants import *
from taproot.tasks.base import Task
from taproot.util import to_pil_array

if TYPE_CHECKING:
    from PIL.Image import Image
    from taproot.hinting import ImageType

__all__ = ["ImageSimilarity"]

MAX_MSE = 255 ** 2
MAX_PSNR = 100

class ImageSimilarity(Task):
    """
    Uses various traditional (non-AI) measures to compare two images.
    All methods return a similarity score between 0 and 1, where 1 means the images are identical.
    """
    task = "image-similarity"
    default = True
    display_name = "Traditional Image Similarity"

    """Authorship Metadata"""
    author = "Benjamin Paine"
    author_url = "https://github.com/painebenjamin/taproot"
    author_affiliations = ["Taproot"]

    """License Metadata"""
    license = LICENSE_APACHE

    @classmethod
    def required_packages(cls) -> Dict[str, Optional[str]]:
        """
        Returns the required packages for this task.
        """
        return {
            "cv2": OPENCV_VERSION_SPEC,
            "numpy": NUMPY_VERSION_SPEC,
            "skimage": SKIMAGE_VERSION_SPEC,
        }

    def mean_squared_error(self, left: Image, right: Image) -> float:
        """
        Calculates the mean-squared error (MSE) between two images.
        """
        import numpy as np
        mse_error = np.sum((np.array(left).astype("float") - np.array(right).astype("float")) ** 2)
        mse_error /= float(left.size[0] * left.size[1])
        return float(1 - (mse_error / MAX_MSE))

    def structural_similarity_index(self, left: Image, right: Image) -> float:
        """
        Calculates the structural similarity index (SSIM) between two images.
        """
        import numpy as np
        from skimage.metrics import structural_similarity as ssim # type: ignore[import-not-found,import-untyped,unused-ignore]
        left_array = np.array(left)
        right_array = np.array(right)
        data_range = right_array.max() - right_array.min()
        value: float = ssim(left_array, right_array, data_range=data_range, multichannel=True, channel_axis=2) # type: ignore[no-untyped-call,unused-ignore]
        return (value + 1) / 2 # [0, 1]

    def peak_signal_to_noise_ratio(self, left: Image, right: Image) -> float:
        """
        Calculates the peak signal-to-noise ratio (PSNR) between two images.
        """
        from math import log10, sqrt
        mse_error = (1 - self.mean_squared_error(left, right)) * MAX_MSE
        if mse_error == 0:
            return 1.0
        psnr_value = 20 * log10(255 / sqrt(mse_error))
        return psnr_value / MAX_PSNR

    def histogram_correlation(self, left: Image, right: Image) -> float:
        """
        Calculates the histogram correlation between two images.
        """
        import cv2 # type: ignore[import-not-found,unused-ignore]
        import numpy as np
        left_hist = cv2.calcHist([np.array(left)], [0], None, [256], [0, 256])
        right_hist = cv2.calcHist([np.array(right)], [0], None, [256], [0, 256])
        correlation: float = cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CORREL) # [-1, 1]
        return (correlation + 1) / 2 # [0, 1]

    def features_similarity(self, left: Image, right: Image, keep_ratio: float=0.3) -> float:
        """
        Calculates the similarity between two images based on their features.
        """
        import cv2 # type: ignore[import-not-found,unused-ignore]
        import numpy as np
        # Convert images to grayscale
        left_gray = cv2.cvtColor(np.array(left), cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(np.array(right), cv2.COLOR_BGR2GRAY)

        # Initiate ORB detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with ORB
        left_kp, left_des = orb.detectAndCompute(left_gray, None)
        right_kp, right_des = orb.detectAndCompute(right_gray, None)

        # Match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(left_des, right_des)
        matches = sorted(matches, key=lambda x: x.distance)

        # Keep only the top matches
        num_matches = int(len(matches) * keep_ratio)
        matches = matches[:num_matches]

        # Calculate the similarity
        total_keypoints = min(len(left_kp), len(right_kp))
        if total_keypoints == 0:
            return 0.0
        return len(matches) / int(total_keypoints * keep_ratio)

    def standardize_images(self, image: ImageType) -> List[Image]:
        """
        Standardizes the input images to the same size and mode.
        """
        images = to_pil_array(image)
        image_sizes = [image.size for image in images]
        image_width = min(image_sizes, key=lambda x: x[0])[0]
        image_height = min(image_sizes, key=lambda x: x[1])[1]
        return [
            image.crop((0, 0, image_width, image_height)).convert("RGB")
            for image in images
        ]

    def __call__( # type: ignore[override]
        self,
        *,
        left: ImageType,
        right: ImageType,
        method: Literal["mse", "ssim", "psnr", "histogram", "features"] = "mse",
    ) -> Union[float, List[float]]:
        """
        Calculates all similarity measures between two or more images.
        When two arrays are passed, assume these should be compared pairwise.

        :param left: The left image or list of images.
        :param right: The right image or list of images.
        :param method: The similarity method to use.
        :return: The similarity score or list of similarity scores.
        """
        left_images = self.standardize_images(left)
        right_images = self.standardize_images(right)
        num_images = min(len(left_images), len(right_images))
        compare_pairwise = num_images > 1

        # Get the method to use
        method = method.lower() # type: ignore[assignment]
        if method == "mse":
            similarity_method = self.mean_squared_error
        elif method == "ssim":
            similarity_method = self.structural_similarity_index
        elif method == "psnr":
            similarity_method = self.peak_signal_to_noise_ratio
        elif method == "histogram":
            similarity_method = self.histogram_correlation
        elif method == "features":
            similarity_method = self.features_similarity
        else:
            raise ValueError(f"Unknown similarity method: {method}")

        # Calculate the similarity
        if compare_pairwise:
            return [
                similarity_method(left_image, right_image)
                for left_image, right_image in zip(left_images, right_images)
            ]
        else:
            return similarity_method(left_images[0], right_images[0])
