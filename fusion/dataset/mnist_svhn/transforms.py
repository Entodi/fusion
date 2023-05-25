from torch import Tensor
from torchvision import transforms


class SVHNTransform:
    """ """

    def __call__(self, x) -> Tensor:
        """
        Make SVHN transform

        Args:
            x: Input tensor
            
        Returns:
            Transform tensor
        """
        x = transforms.ToTensor()(x)
        return {"1": x}


class MNISTTransform:
    """ """

    def __call__(self, x) -> Tensor:
        """
        Make MNIST transform
        
        Args:
            x: Input tensor

        Returns:
            Transform tensor
        """
        x = transforms.Resize((32, 32))(x)
        x = transforms.ToTensor()(x)
        return {"0": x}
