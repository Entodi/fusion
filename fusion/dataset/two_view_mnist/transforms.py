import torch
from torch import Tensor
from torchvision import transforms

from fusion.dataset.abasetransform import ABaseTransform


class UnitIntervalScale(ABaseTransform):
    def __call__(self, x):
        """
        Make  Unit Interval Scale transform

        Args:
            x: Input tensor
            
        Returns:
            Transform tensor
        """

        x = (x - x.min()) / (x.max() - x.min())
        return x


class RandomRotation(ABaseTransform):
    def __init__(self, degrees: int = 45):
        """
        Initialization  Class Random Rotation transform

        Args:
            degrees: Max angle

        Returns:
            Class Random Rotation transform
        """
        self.random_rotation = transforms.RandomRotation(degrees, fill=(0,))

    def __call__(self, x):
        """
        Make  Random Rotation transform

        Args:
            x: Input tensor

        Returns:
            Transform tensor
        """
        x = self.random_rotation(x)
        x = transforms.ToTensor()(x)
        return x


class UniformNoise(ABaseTransform):
    def __call__(self, x) -> Tensor:
        """
        Make  Uniform Noise transform

        Args:
            x: Input tensor

        Returns:
            Transform tensor
        """
        x = transforms.ToTensor()(x)
        x = x + torch.rand(x.size())
        x = torch.clamp(x, min=0.0, max=1.0)
        return x


class TwoViewMnistTransform(ABaseTransform):
    def __init__(self, keys=["0", "1"]):
        self._keys = keys

    def __call__(self, x):
        """
        Make  Two View Mnist transform

        Args:
            x: Input tensor

        Returns:
            Transform tensor
        """
        x = transforms.ToTensor()(x)
        x = UnitIntervalScale()(x)
        x = transforms.ToPILImage()(x)
        x = transforms.Resize((32, 32))(x)
        v1 = RandomRotation()(x)
        v2 = UniformNoise()(x)
        batch = {}
        for target, source in self._keys.items():
            target = target.split('_')[0]
            if target == "0":
                batch[source] = v1
            elif target == "1":
                batch[source] = v2
        return batch


class RandomRotationTransform(ABaseTransform):
    def __init__(self, key="0"):
        self._key = key

    def __call__(self, x):
        """
        Make  Random Rotation transform

        Args:
            x: Input tensor

        Returns:
             Transform tensor
        """
        x = transforms.ToTensor()(x)
        x = UnitIntervalScale()(x)
        x = transforms.ToPILImage()(x)
        x = transforms.Resize((32, 32))(x)
        x = RandomRotation()(x)
        return {self._key: x}


class UniformNoiseTransform(ABaseTransform):
    def __init__(self, key="1"):
        self._key = key

    def __call__(self, x):
        """
        Make  Uniform Noise transform

        Args:
            x: Input tensor
            
        Returns:
             Transform tensor
        """
        x = transforms.ToTensor()(x)
        x = UnitIntervalScale()(x)
        x = transforms.ToPILImage()(x)
        x = transforms.Resize((32, 32))(x)
        x = UniformNoise()(x)
        return {self._key: x}
