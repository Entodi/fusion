from lib2to3.pgen2.token import OP
from optparse import Option
from typing import Any, Dict, Optional, Tuple

import torch.nn as nn
from torch import Tensor

from fusion.architecture.abasearchitecture import (
    ABaseArchitecture,
    TActivation,
    TDropout,
    TConv,
    TNorm,
    TPool
)


class BaseConvLayer(ABaseArchitecture):
    def __init__(
        self,
        conv_layer_class: TConv,
        conv_layer_args: Dict[str, Any],
        norm_layer_class: Optional[TNorm] = None,
        norm_layer_args: Dict[str, Any] = {},
        dp_layer_class: Optional[TDropout] = None,
        dp_layer_args: Dict[str, Any] = {},
        activation_class: Optional[TActivation] = None,
        activation_args: Dict[str, Any] = {},
        pool_layer_class: Optional[TPool] = None,
        pool_layer_args: Dict[str, Any] = {},
        input_dim: int = 2,
        weights_initialization_type: str = "xavier_uniform",
    ):
        """
        Base class of the convolution layer, this class allows the specification of the 
        following sub-layers in order of appearing in the forward function:
            1) A convolutional layer
            2) A normalization method
            3) A dropout layer
            4) The non-linear activation that should be used
        The forward function in the class returns a tuple: (output after all specified sub-layers, 
        output after convolutional sub-layer only)

        Args:
            conv_layer_class: Convolutional layer class
            conv_layer_args: Parameters for the convolution layer
            norm_layer_class:  Normalization layer class
            norm_layer_args: Parameters for the normalization layer
            dp_layer_class: Dropout layer class
            dp_layer_args: Parameters for the droupout layer
            pool_layer_class: Pooling layer class
            pool_layer_args: Pooling layer args
            activation_class: Activation function class
            activation_args: Parameters for the activation function
            weights_initialization_type: Type of initialization weights

        Returns:
            Base class of convolution layer
        """

        super(BaseConvLayer, self).__init__(
            input_dim=input_dim,
            conv_layer_class=conv_layer_class,
            norm_layer_class=norm_layer_class,
            dp_layer_class=dp_layer_class,
            activation_class=activation_class,
            pool_layer_class=pool_layer_class,
            weights_initialization_type=weights_initialization_type,
        )
        self._layer = nn.ModuleList()
        # print (conv_layer_args)
        self._layer.append(self._conv_layer_class(**conv_layer_args))
        if self._norm_layer_class:
            self._layer.append(self._norm_layer_class(**norm_layer_args))
        if self._dp_layer_class:
            self._layer.append(self._dp_layer_class(**dp_layer_args))
        if self._pool_layer_class:
            self._layer.append(self._pool_layer_class(**pool_layer_args))
        if self._activation_class:
            self._layer.append(self._activation_class(**activation_args))
        # self.init_weights()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward method for the base class of this custom convolutional layer

        Args:
            x: Input tensor

        Returns:
            x: The tensor after passing through all the specified sub-layers: convolutional layer, 
            normalization layer, dropout layer, activation function 
            conv_latent: The tensor after only passing through the convolutional sub-layer
        """
        x = self._layer[0](x)
        conv_latent = x
        for layer in self._layer[1:]:
            x = layer(x)
        return x, conv_latent

    def init_weights(self, gain: float = nn.init.calculate_gain('relu')):
        """
        Method for initialization weights
        
        Returns:
            Layer with initialization weights

        """
        if self._norm_layer_class:
            if isinstance(self._layer[1], self._norm_layer_class):
                self._layer[1].weight.data.fill_(1)
                self._layer[1].bias.data.zero_()
        if self._weights_initialization_type == "xavier_uniform":
            nn.init.xavier_uniform_(
                self._layer[0].weight, gain=gain
            )
            if not isinstance(self._layer[0].bias, type(None)):
                nn.init.constant_(self._layer[0].bias, 0)
        elif self._weights_initialization_type == "skip":
            pass
        else:
            raise NotImplementedError
