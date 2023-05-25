import nibabel as nib
import numpy as np
import torch
from torchio.transforms import Transform


class MNIMaskTransform(Transform):
    def __init__(self, template="MNI152_T1_3mm_brain_mask_dil_cubic192.nii.gz", p=1):
        super(MNIMaskTransform, self).__init__(p=p)
        self._mask = ~torch.BoolTensor(nib.load(template).get_fdata()).unsqueeze(0)

    def apply_transform(self, volume):
        for key, vol in volume.items():
            if "source" in key:
                volume[key]["data"] = self.process_volume(vol["data"])
        return volume

    def process_volume(self, volume):
        temp = volume
        volume[self._mask] = 0
        return volume


class VolumetricRandomCrop(Transform):
    def __init__(self, shape, p=1):
        super().__init__(p=p)
        self.shape = shape

    def apply_transform(self, volume):
        start, end = None, None
        for key, vol in volume.items():
            if "source" in key:
                volume[key]["data"], start, end = self.process_volume(
                    vol["data"], start, end
                )
        return volume

    def process_volume(self, volume, start=None, end=None):
        if start is not None and end is not None:
            return (
                volume[:, start[0] : end[0], start[1] : end[1], start[2] : end[2]],
                start,
                end,
            )
        else:
            start = np.random.randint(0, high=volume.shape[1] - self.shape, size=3)
            end = start + self.shape
            return (
                volume[:, start[0] : end[0], start[1] : end[1], start[2] : end[2]],
                start,
                end,
            )
