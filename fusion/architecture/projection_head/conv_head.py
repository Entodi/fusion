import torch
import torch.nn as nn
from torch import Tensor

from fusion.architecture import ABaseArchitecture
from fusion.architecture.abasearchitecture import TActivation, TConv, TNorm
from fusion.architecture.base_block import BaseConvLayer


class ConvHead(ABaseArchitecture):
    def __init__(
        self,
        dim_in: int,
        dim_l: int,
        dim_h: int,
        num_h_layers: int = 1,
        input_dim: int = 2,
        conv_layer_class: TConv = nn.Conv2d,
        norm_layer_class: TNorm = nn.BatchNorm2d,
        activation_class: TActivation = nn.ReLU,
        weights_initialization_type: str = "xavier_uniform",
        use_bias: bool = False,
    ):
        """
        Initialization of Convolution head model

        Args:
            dim_in: The number of input channels
            dim_l: The number of latent dimensions
            dim_h: The number of feature channels for the convolutional layer. It is kept fixed for all hidden layers
            num_h_layers: The number of convolutional layers
            conv_layer_class: The type of convolutional layer to use, default=nn.Conv2d
            norm_layer_class: The type of normalization layer to use, default=nn.BatchNorm2d
            activation_class: The type of non-linear activation function to use, default=nn.LeakyReLU
            weights_initialization_type: The weight initialization type to use, default='xavier_uniform'
            use_bias: Flag of use bias in convolutional layer

        Returns:
            Convolution head model
        """
        super().__init__(
            input_dim=input_dim,
            conv_layer_class=conv_layer_class,
            norm_layer_class=norm_layer_class,
            activation_class=activation_class,
            weights_initialization_type=weights_initialization_type,
        )
        self._dim_in = dim_in
        self._dim_l = dim_l
        self._bn_embedding = self._norm_layer_class(dim_l, affine=True)
        self._convolutional_path = nn.ModuleList([])
        # add first layer
        self._convolutional_path.append(
            BaseConvLayer(
                conv_layer_class,
                {
                    "in_channels": self._dim_in,
                    "out_channels": dim_h,
                    "kernel_size": 1,
                    "bias": use_bias,
                },
                input_dim=self._input_dim,
                norm_layer_class=self._norm_layer_class,
                norm_layer_args={"num_features": dim_h},
                activation_class=self._activation_class,
                activation_args={"inplace": True},
                weights_initialization_type=self._weights_initialization_type,
            )
        )
        for i in range(1, num_h_layers):
            self._convolutional_path.append(
                BaseConvLayer(
                    conv_layer_class,
                    {
                        "in_channels": dim_h,
                        "out_channels": dim_h,
                        "kernel_size": 1,
                        "bias": use_bias,
                    },
                    input_dim=self._input_dim,
                    norm_layer_class=self._norm_layer_class,
                    norm_layer_args={"num_features": dim_h},
                    activation_class=self._activation_class,
                    activation_args={"inplace": True},
                    weights_initialization_type=self._weights_initialization_type,
                )
            )
        # add last layer
        self._convolutional_path.append(
            BaseConvLayer(
                conv_layer_class,
                {
                    "in_channels": dim_h,
                    "out_channels": self._dim_l,
                    "kernel_size": 1,
                    "bias": use_bias,
                },
                input_dim=self._input_dim,
                weights_initialization_type=self._weights_initialization_type,
            )
        )
        self._identity_shortcut = BaseConvLayer(
            conv_layer_class,
            {
                "in_channels": dim_in,
                "out_channels": dim_l,
                "kernel_size": 1,
                "bias": use_bias,
            },
            input_dim=self._input_dim,
            weights_initialization_type="skip",
        )

    def init_weights(self):
        """
        Method for initialization weights

        Returns:
            Convolution head model with initialization weights
        """
        # initialization of the convolutional path
        for layer in self._convolutional_path:
            layer.init_weights()
        # initialization of identity path
        # according to AMDIM implementation
        # https://github.com/Philip-Bachman/amdim-public/blob/8754ae149ed28da8066f696f95ba4ca0e3ffebd8/model.py#L392
        # initialize shortcut to be like identity (if possible)
        if self._dim_l >= self._dim_in:
            eye_mask = None
            if self._conv_layer_class is nn.Conv3d:
                eye_mask = torch.zeros(self._dim_l, self._dim_in, 1, 1, 1, dtype=bool)
                for i in range(self._dim_in):
                    eye_mask[i, i, 0, 0, 0] = 1
            elif self._conv_layer_class is nn.Conv2d:
                eye_mask = torch.zeros(self._dim_l, self._dim_in, 1, 1, dtype=bool)
                for i in range(self._dim_in):
                    eye_mask[i, i, 0, 0] = 1
            else:
                raise NotImplementedError
            self._identity_shortcut._layer[0].weight.data.uniform_(-0.01, 0.01)
            self._identity_shortcut._layer[0].weight.data.masked_fill_(eye_mask, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward method of Convolution head model

        Args:
            x:  input tensor
            
        Returns:
            x
        """
        identity, _ = self._identity_shortcut(x)
        for layer in self._convolutional_path:
            x, _ = layer(x)
        x = self._bn_embedding(x + identity)
        return x
