from numpy import isin
import torch.nn as nn
from torch import Tensor

from fusion.architecture import ABaseArchitecture
from fusion.architecture.abasearchitecture import TActivation, TConv, TNorm, TDropout, TPool
from fusion.architecture.base_block import BaseConvLayer, Flatten


class AneesAlexNet(ABaseArchitecture):
    def __init__(
        self,
        dim_in: int,
        dim_l: int,
        input_size: int = 128,
        input_dim: int = 2,
        conv_layer_class: TConv = nn.Conv2d,
        norm_layer_class: TNorm = nn.BatchNorm2d,
        dp_layer_class: TDropout = nn.Dropout2d,
        activation_class: TActivation = nn.ReLU,
        pool_layer_class: TPool = nn.MaxPool2d,
        weights_initialization_type: str = "xavier_uniform",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            conv_layer_class=conv_layer_class,
            norm_layer_class=norm_layer_class,
            dp_layer_class=dp_layer_class,
            pool_layer_class=pool_layer_class,
            activation_class=activation_class,
            weights_initialization_type=weights_initialization_type,
        )
        assert input_size == 128, 'Currently parameters are hard-coded'
        self._dim_in = dim_in
        self._dim_l = dim_l
        self._conv_layers: nn.ModuleList
        self._linear_layers: nn.Sequential
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
                        "out_channels": 128,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
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
                )
            ]
        )
        self._linear_layers = nn.Sequential(
            Flatten(),
            #nn.Dropout(),
            nn.Linear(128 * 2 ** self._input_dim, self._dim_l, bias=False),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
        )


    def init_weights(self):
        for layer in self._layers:
            layer.init_weights(gain=nn.init.calculate_gain('relu'))
        for m in self._linear_layers.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: Tensor) -> Tensor:
        #print (x.size())
        for layer in self._layers:
            x, _ = layer(x)
            #print (x.size())
        x = self._linear_layers(x)
        return x, None
