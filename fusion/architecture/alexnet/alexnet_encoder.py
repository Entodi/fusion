import torch.nn as nn
from torch import Tensor

from fusion.architecture import ABaseArchitecture
from fusion.architecture.abasearchitecture import TActivation, TConv, TNorm, TDropout, TPool
from fusion.architecture.base_block import BaseConvLayer, Flatten


class AlexNetEncoder(ABaseArchitecture):
    def __init__(
        self,
        dim_in: int,
        dim_l: int,
        dim_cls=None,
        input_size: int = 128,
        input_dim: int = 2,
        conv_layer_class: TConv = nn.Conv2d,
        norm_layer_class: TNorm = nn.BatchNorm2d,
        dp_layer_class: TDropout = nn.Dropout2d,
        activation_class: TActivation = nn.ReLU,
        pool_layer_class: TPool = nn.MaxPool2d,
        weights_initialization_type: str = "xavier_uniform",
    ):
        super().__init__(
            input_dim=input_dim,
            conv_layer_class=conv_layer_class,
            norm_layer_class=norm_layer_class,
            dp_layer_class=dp_layer_class,
            pool_layer_class=pool_layer_class,
            activation_class=activation_class,
            weights_initialization_type=weights_initialization_type,
        )
        self._dim_in = dim_in
        self._dim_l = dim_l
        self._dim_cls = dim_cls
        self._input_size = input_size
        self._flatten = Flatten()
        self._layers: nn.ModuleList
        self._construct()

    def _construct(self):
        self._layers = nn.ModuleList(
            [
                BaseConvLayer(
                    self._conv_layer_class,
                    {
                        "in_channels": self._dim_in,
                        "out_channels": 64,
                        "kernel_size": 5,
                        "stride": 2,
                        "padding": 0,
                        "bias": False
                    },
                    norm_layer_class=self._norm_layer_class,
                    norm_layer_args={"num_features": 64},
                    pool_layer_class=self._pool_layer_class,
                    pool_layer_args={
                        "kernel_size": 3,
                        "stride": 3
                    },
                    activation_class=self._activation_class,
                    activation_args={
                        "inplace": True
                    },
                    input_dim=self._input_dim,
                ),
                BaseConvLayer(
                    self._conv_layer_class,
                    {
                        "in_channels": 64,
                        "out_channels": 128,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 0,
                        "bias": False
                    },
                    norm_layer_class=self._norm_layer_class,
                    norm_layer_args={"num_features": 128},
                    pool_layer_class=self._pool_layer_class,
                    pool_layer_args={
                        "kernel_size": 3,
                        "stride": 3
                    },
                    activation_class=self._activation_class,
                    activation_args={
                        "inplace": True
                    },
                    input_dim=self._input_dim,
                ),
                BaseConvLayer(
                    self._conv_layer_class,
                    {
                        "in_channels": 128,
                        "out_channels": 192,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                        "bias": False
                    },
                    norm_layer_class=self._norm_layer_class,
                    norm_layer_args={"num_features": 192},
                    activation_class=self._activation_class,
                    activation_args={
                        "inplace": True
                    },
                    input_dim=self._input_dim,
                ),
                BaseConvLayer(
                    self._conv_layer_class,
                    {
                        "in_channels": 192,
                        "out_channels": 192,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                        "bias": False
                    },
                    norm_layer_class=self._norm_layer_class,
                    norm_layer_args={"num_features": 192},
                    activation_class=self._activation_class,
                    activation_args={
                        "inplace": True
                    },
                    input_dim=self._input_dim,
                ),
                BaseConvLayer(
                    self._conv_layer_class,
                    {
                        "in_channels": 192,
                        "out_channels": self._dim_l,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                        "bias": False
                    },
                    #norm_layer_class=self._norm_layer_class,
                    #norm_layer_args={"num_features": self._dim_l},
                    pool_layer_class=nn.AdaptiveAvgPool3d if self._input_dim == 3 else nn.AdaptiveAvgPool2d,
                    pool_layer_args={
                        "output_size": 1
                    },
                    #activation_class=self._activation_class,
                    #activation_args={
                    #    "inplace": True
                    #},
                    input_dim=self._input_dim,
                )
            ]
        )

    def init_weights(self):
        for layer in self._layers:
            layer.init_weights(gain=nn.init.calculate_gain('relu'))

    def forward(self, x: Tensor) -> Tensor:
        #print (self._dim_cls)
        latents = None
        if self._dim_cls is not None:
            latents = {}
        for counter, layer in enumerate(self._layers):
            x, conv_latent = layer(x)
            #print (x.size(), conv_latent.size())
            # Add conv latent
            if self._dim_cls is not None:
                #print (conv_latent.size()[-1])
                if f'{conv_latent.size()[-1]}_{counter}' in self._dim_cls:
                    latents[f'{conv_latent.size()[-1]}_{counter}'] = conv_latent
                    counter += 1
        if self._dim_cls is not None:
            latents[f'1_{counter}'] = x
        z = self._flatten(x)
        return z, latents
