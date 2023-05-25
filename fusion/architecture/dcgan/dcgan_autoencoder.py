from .dcgan_encoder import DcganEncoder
from .dcgan_decoder import DcganDecoder

from typing import Tuple

import torch.nn as nn
from torch import Tensor

from fusion.architecture import ABaseArchitecture
from fusion.architecture.abasearchitecture import TActivation, TConv, TNorm


class DcganAutoEncoder(ABaseArchitecture):
    def __init__(
        self,
        dim_in: int,
        dim_h: int,
        dim_l: int,
        dim_cls=None,
        input_size: int = 32,
        input_dim: int = 2,
        conv_layer_class: TConv = nn.Conv2d,
        conv_t_layer_class: TConv = nn.ConvTranspose2d,
        norm_layer_class: TNorm = nn.BatchNorm2d,
        activation_class: TActivation = nn.LeakyReLU,
        weights_initialization_type: str = "xavier_uniform",
    ):
        """
        The DCGAN Autoencoder class

        Args:
            dim_in: The number of input channels
            dim_h: The number of feature channels for the first convolutional layer, the number of feature channels double with each next convolutional layer in the encoder
                          The number of feature channels are consecutively halved in the decoder starting with the first and the last layer has dim_h number of feature channels
            dim_l: The number of latent dimensions
            dim_cls: A list of scalars, where each number should correspond to the output width for one of the convolutional layers.
                            The information between latent variable z and the convolutional feature maps width widths in dim_cls are maximized.
                            If dim_cls=None, the information between z and none of the convolutional feature maps is maximized, default=None
            input_size: The input width and height of the image, default=32
            input_dim: The number of input dimensions, e.g. an image is 2-dimensional (input_dim=2) and a volume is 3-dimensional (input_dim=3), default=2
            conv_layer_class: The type of convolutional layer to use, default=nn.Conv2d
            conv_t_layer_class: The type of transposed convolutional layer to use, default=nn.ConvTranspose2d
            norm_layer_class: The type of normalization layer to use, default=nn.BatchNorm2d
            activation_class: The type of non-linear activation function to use, default=nn.LeakyReLU
            weights_initialization_type: The weight initialization type to use, default='xavier_uniform'

        Returns:
            Class of DCGAN autoencoder model

        """
        super().__init__()
        self._encoder = DcganEncoder(
            dim_in,
            dim_h,
            dim_l,
            dim_cls=dim_cls,
            input_size=input_size,
            input_dim=input_dim,
            conv_layer_class=conv_layer_class,
            norm_layer_class=norm_layer_class,
            activation_class=activation_class,
            weights_initialization_type=weights_initialization_type,
        )
        self._decoder = DcganDecoder(
            dim_in,
            dim_h,
            dim_l,
            dim_cls=dim_cls,
            input_size=input_size,
            input_dim=input_dim,
            conv_layer_class=conv_t_layer_class,
            norm_layer_class=norm_layer_class,
            activation_class=activation_class,
            weights_initialization_type=weights_initialization_type,
        )

    def forward(self, x: Tensor) -> Tuple[Tuple, Tuple]:
        """
        The forward method for the DCGAN autoencoder model

        Args:
            x: An input tensor
            
        Returns:
            z: The latent variable
            x_hat: A reconstruction of the original input tensor
        """
        z, _ = self._encoder(x)
        x_hat, _ = self._decoder(z)
        return z, x_hat

    def init_weights(self):
        """
        The weight initialization method for the encoder and decoder in the autoencoder

        Returns:
            Autoencoder with initialized weights
        """
        self._encoder.init_weights()
        self._decoder.init_weights()
