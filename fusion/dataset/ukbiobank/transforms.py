import numpy as np
import scipy.ndimage as ndi
from typing import Tuple
from typing import Union
from typing import List

import torch

from torchio import Subject
from torchio.transforms.augmentation import RandomTransform
from torchio.transforms import IntensityTransform
from torchio.typing import TypeSextetFloat, TypeTripletFloat, TypeData


class CustomRandomBlur(RandomTransform, IntensityTransform):
    def __init__(
        self,
        skip_sources: List[int] = [2],
        std: Union[float, Tuple[float, float]] = (0, 2),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.std_ranges = self.parse_params(std, None, 'std', min_constraint=0)
        self._skip_sources = skip_sources

    def apply_transform(self, subject: Subject) -> Subject:
        for name, data in self.get_images_dict(subject).items():
            if "source" in name:
                source_id = int(name.split("_")[-1])
                if subject[f"pipeline_{source_id}"] not in self._skip_sources:
                    std = self.get_params(self.std_ranges)
                    transformed = self.apply_blur(data, std)
                    data.set_data(transformed)
        return subject

    def get_params(self, std_ranges: TypeSextetFloat) -> TypeTripletFloat:
        std = self.sample_uniform_sextet(std_ranges)
        return std

    def apply_blur(self, image, stds):
        repets = image.num_channels, 1
        stds_channels: np.ndarray
        stds_channels = np.tile(stds, repets)  # type: ignore[arg-type]
        transformed_tensors = []
        for std, channel in zip(stds_channels, image.data):
            transformed_tensor = self.blur(
                channel,
                image.spacing,
                std,
            )
            transformed_tensors.append(transformed_tensor)
        return torch.stack(transformed_tensors)

    @staticmethod
    def blur(
        data: TypeData,
        spacing: TypeTripletFloat,
        std_physical: TypeTripletFloat,
    ) -> torch.Tensor:
        assert data.ndim == 3
        # For example, if the standard deviation of the kernel is 2 mm and the
        # image spacing is 0.5 mm/voxel, the kernel should be
        # (2 mm / 0.5 mm/voxel) = 4 voxels wide
        std_voxel = np.array(std_physical) / np.array(spacing)
        blurred = ndi.gaussian_filter(data, std_voxel)
        tensor = torch.as_tensor(blurred)
        return tensor
