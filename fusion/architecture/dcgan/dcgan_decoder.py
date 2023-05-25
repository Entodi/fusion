from typing import Dict, Tuple, Type

from torch import Tensor
import torch.nn as nn

from fusion.architecture import ABaseArchitecture
from fusion.architecture.base_block import BaseConvLayer, Unflatten


class DcganDecoder(ABaseArchitecture):
    def __init__(
        self,
        dim_in: int,
        dim_h: int,
        dim_l: int,
        dim_cls=None,
        input_size: int = 32,
        input_dim: int = 2,
        conv_layer_class: Type[nn.modules.conv._ConvNd] = nn.ConvTranspose2d,
        norm_layer_class: Type[nn.modules.batchnorm._BatchNorm] = nn.BatchNorm2d,
        activation_class: Type[nn.Module] = nn.ReLU,
        weights_initialization_type: str = "xavier_uniform",
    ):
        """
        Class of DCGAN Decoder

        Args:
            dim_in: The number of input channels
            dim_h: The number of feature channels for the last transposed convolutional layer,
                          the number of feature channels are halved after for each consecutive transposed convolutional layer after the first
            dim_l: The number of latent dimensions
            dim_cls: A list of scalars, where each number should correspond to the output width for one of the convolutional layers.
                             The information between latent variable z and the convolutional feature maps width widths in dim_cls are maximized.
                             If dim_cls=None, the information between z and none of the convolutional feature maps is maximized, default=None
            input_size: The input width and height of the image, default=32
            input_dim: The number of input dimensions, e.g. an image is 2-dimensional (input_dim=2) and a volume is 3-dimensional (input_dim=3), default=2
            conv_layer_class: The type of transposed convolutional layer to use, default=nn.ConvTranspose2d
            norm_layer_class: The type of normalization layer to use, default=nn.BatchNorm2d
            activation_class: The type of non-linear activation function to use, default=nn.ReLU
            weights_initialization_type: The weight initialization type to use, default='xavier_uniform'

        Returns:
            Class of DCGAN decoder model
        """
        super().__init__(
            input_dim=input_dim,
            conv_layer_class=conv_layer_class,
            norm_layer_class=norm_layer_class,
            activation_class=activation_class,
            weights_initialization_type=weights_initialization_type,
        )
        self._parse()
        self._dim_in = dim_in
        self._dim_h = dim_h
        self._dim_l = dim_l
        self._dim_cls = dim_cls
        self._input_size = input_size
        self._unflatten = Unflatten(input_dim=input_dim)
        self._layers: nn.ModuleList = nn.ModuleList([])
        self._construct()

    def _construct(self):
        if self._input_size == 64:
            self._layers.append(
                BaseConvLayer(
                    self._conv_layer_class,
                    {
                        "in_channels": self._dim_l,
                        "out_channels": 8 * self._dim_h,
                        "kernel_size": 4,
                        "stride": 1,
                        "padding": 0,
                        "bias": False,
                    },
                    norm_layer_class=self._norm_layer_class,
                    norm_layer_args={"num_features": 8 * self._dim_h},
                    activation_class=self._activation_class,
                    activation_args={"inplace": True},
                )
            )
            self._layers.append(
                BaseConvLayer(
                    self._conv_layer_class,
                    {
                        "in_channels": 8 * self._dim_h,
                        "out_channels": 4 * self._dim_h,
                        "kernel_size": 4,
                        "stride": 2,
                        "padding": 1,
                        "bias": False,
                    },
                    norm_layer_class=self._norm_layer_class,
                    norm_layer_args={"num_features": 4 * self._dim_h},
                    activation_class=self._activation_class,
                    activation_args={"inplace": True},
                )
            )
        elif self._input_size == 32:
            self._layers.append(
                BaseConvLayer(
                    self._conv_layer_class,
                    {
                        "in_channels": self._dim_l,
                        "out_channels": 4 * self._dim_h,
                        "kernel_size": 4,
                        "stride": 1,
                        "padding": 0,
                        "bias": False,
                    },
                    norm_layer_class=self._norm_layer_class,
                    norm_layer_args={"num_features": 4 * self._dim_h},
                    activation_class=self._activation_class,
                    activation_args={"inplace": True},
                )
            )
        else:
            raise NotImplementedError(
                "DCGAN only supports input square images ' + \
                'with size 32, 64 in current implementation."
            )

        self._layers.append(
            BaseConvLayer(
                self._conv_layer_class,
                {
                    "in_channels": 4 * self._dim_h,
                    "out_channels": 2 * self._dim_h,
                    "kernel_size": 4,
                    "stride": 2,
                    "padding": 1,
                    "bias": False,
                },
                norm_layer_class=self._norm_layer_class,
                norm_layer_args={"num_features": 2 * self._dim_h},
                activation_class=self._activation_class,
                activation_args={"inplace": True},
            )
        )
        self._layers.append(
            BaseConvLayer(
                self._conv_layer_class,
                {
                    "in_channels": 2 * self._dim_h,
                    "out_channels": self._dim_h,
                    "kernel_size": 4,
                    "stride": 2,
                    "padding": 1,
                    "bias": False,
                },
                norm_layer_class=self._norm_layer_class,
                norm_layer_args={"num_features": self._dim_h},
                activation_class=self._activation_class,
                activation_args={"inplace": True},
            )
        )
        self._layers.append(
            BaseConvLayer(
                self._conv_layer_class,
                {
                    "in_channels": self._dim_h,
                    "out_channels": self._dim_in,
                    "kernel_size": 4,
                    "stride": 2,
                    "padding": 1,
                    "bias": False,
                },
                #activation_class=nn.Tanh,
                activation_class=nn.Sigmoid,
            )
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[int, Tensor]]:
        """
        The forward method for the DCGAN autoencoder model

        Args:
            x: The input tensor

        Returns:
            x_hat: A reconstruction of the original input tensor
            latents: The convolutional feature maps, with widths specified by self._dim_cls
        """
        x_hat = self._unflatten(x)
        latents = None
        # Adds latent
        if self._dim_cls is not None:
            latents = {}
            latents[1] = x_hat
        for layer in self._layers:
            x_hat, conv_latent = layer(x_hat)
            # Add conv latent
            if self._dim_cls is not None:
                if conv_latent.size()[-1] in self._dim_cls:
                    latents[conv_latent.size()[-1]] = conv_latent
        return x_hat, latents

    def init_weights(self):
        """
        Weight initialization method

        Returns:
            DcganDecoder with initialized weights

        """
        for layer in self._layers[:-1]:
            layer.init_weights()
        self._layers[-1].init_weights(nn.init.calculate_gain('sigmoid'))

    def _parse(self):
        if isinstance(self._conv_layer_class, str):
            if self._conv_layer_class == 'ConvTranspose2d':
                self._conv_layer_class = nn.ConvTranspose2d
                assert self._input_dim == 2
            elif self._conv_layer_class == 'ConvTranspose3d':
                self._conv_layer_class = nn.ConvTranspose3d
                assert self._input_dim == 3
            else:
                assert False, 'Only Conv2d and Conv3d are supported'
        if isinstance(self._norm_layer_class, str):
            if self._norm_layer_class == 'BatchNorm2d':
                self._norm_layer_class = nn.BatchNorm2d
                assert self._input_dim == 2
            elif self._norm_layer_class == 'BatchNorm3d':
                self._norm_layer_class = nn.BatchNorm3d
                assert self._input_dim == 3
            else:
                assert False, 'Only BatchNorm2d and BatchNorm3d are supported'
        if isinstance(self._dp_layer_class, str):
            if self._dp_layer_class == 'Dropout2d':
                self._dp_layer_class = nn.Dropout2d
                assert self._input_dim == 2
            elif self._dp_layer_class == 'Dropout3d':
                self._dp_layer_class = nn.Dropout3d
                assert self._input_dim == 3
            else:
                assert False, 'Only Dropout2d and Dropout3d are supported'
