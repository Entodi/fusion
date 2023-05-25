import abc
from typing import Optional, Type

import torch.nn as nn


TActivation = Type[nn.Module]
TDropout = Type[nn.modules.dropout._DropoutNd]
TConv = Type[nn.modules.conv._ConvNd]
TNorm = Type[nn.modules.batchnorm._BatchNorm]
TPool = Type[nn.modules.pooling._MaxPoolNd]


class ABaseArchitecture(abc.ABC, nn.Module):
    @abc.abstractmethod
    def __init__(
        self,
        input_dim: int = 2,
        conv_layer_class: TConv = nn.Conv2d,
        norm_layer_class: TNorm = None,
        dp_layer_class: Optional[TDropout] = None,
        activation_class: Optional[TActivation] = None,
        pool_layer_class: Optional[TPool] = None,
        weights_initialization_type: Optional[str] = None,
    ):
        """
        Args:
            conv_layer_class:
            norm_layer_class:
            dp_layer_class:
            activation_class:
            weights_initialization_type:
        """
        super().__init__()
        self._layers: Optional[nn.ModuleList] = None
        self._input_dim = input_dim
        self._conv_layer_class = conv_layer_class
        self._norm_layer_class = norm_layer_class
        self._dp_layer_class = dp_layer_class
        self._activation_class = activation_class
        self._pool_layer_class = pool_layer_class
        self._weights_initialization_type = weights_initialization_type
        self._parse()

    @abc.abstractmethod
    def init_weights(self):
        """Weight initialization"""
        pass

    def get_layers(self):
        """
        Get layers
        
        Returns: Layers
        """
        return self._layers

    def _parse(self):
        if isinstance(self._conv_layer_class, str):
            if self._conv_layer_class == 'Conv2d':
                self._conv_layer_class = nn.Conv2d
                assert self._input_dim == 2
            elif self._conv_layer_class == 'Conv3d':
                self._conv_layer_class = nn.Conv3d
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
        if isinstance(self._pool_layer_class, str):
            if self._pool_layer_class == 'MaxPool2d':
                assert self._input_dim == 2
                self._pool_layer_class = nn.MaxPool2d
            elif self._pool_layer_class == 'MaxPool3d':
                assert self._input_dim == 3
                self._pool_layer_class = nn.MaxPool3d
            elif self._pool_layer_class == 'AvgPool2d':
                assert self._input_dim == 2
                self._pool_layer_class = nn.AvgPool2d
            elif self._pool_layer_class == 'AvgPool3d':
                assert self._input_dim == 3
                self._pool_layer_class = nn.AvgPool3d
            else:
                assert False, 'Only MaxPool2d and MaxPool3d are supported'
