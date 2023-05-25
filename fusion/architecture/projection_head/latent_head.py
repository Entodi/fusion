from fusion.architecture import ABaseArchitecture
from fusion.architecture.base_block import Flatten

import torch.nn as nn
from torch import Tensor


class LatentHead(ABaseArchitecture):
    def __init__(
        self,
        dim_in: int,
        dim_l: int,
        dim_h: int,
        num_h_layers: int = 2,
        use_linear: bool = False,
        use_bias: bool = False,
        use_bn: bool = True,
    ):
        """
        Initialization Class of Latent Head model

        Args:
            dim_in: The number of input channels
            dim_l: The number of latent dimensions
            dim_h: The number of feature channels for the convolutional layer. It is kept fixed for all hidden layers
            num_h_layers: The number of convolutional layers
            use_linear: Flag of use linear layer
            use_bias: Flag of use bias in convolutional layer
            use_bn: Flag of use batch normalization

        Returns:
            Class of Latent Head model

        """
        super().__init__()
        self._num_h_layers = num_h_layers
        self._use_linear = use_linear
        self._flatten = Flatten()
        head = nn.ModuleList([Flatten()])
        if self._use_linear:
            if self._num_h_layers == 0:
                head.append(nn.Linear(dim_in, dim_l, bias=use_bias))
            else:
                assert dim_h != 0
                # add first hidden layer
                head.append(nn.Linear(dim_in, dim_h, bias=use_bias))
                if use_bn:
                    head.append(nn.BatchNorm1d(dim_h))
                head.append(nn.ReLU(inplace=True))
                # add other self._num_h_layers - 1 layers
                for i in range(1, self._num_h_layers):
                    head.append(nn.Linear(dim_h, dim_h, bias=use_bias))
                    if use_bn:
                        head.append(nn.BatchNorm1d(dim_h))
                    head.append(nn.ReLU(inplace=True))
                # add final layer
                head.append(nn.Linear(dim_h, dim_l, bias=use_bias))
        self._head = nn.Sequential(*head)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward method of Latent Head model
        Args:
            x:  input tensor
            
        Returns:
            x
        """
        if self._use_linear:
            x = self._head(x)
        else:
            x = self._flatten(x)
        return x

    def init_weights(self):
        """
        Method for initialization weights

        Returns:
            Latent Head model with initialization weights
        """
        for layer in self._head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(
                    layer.weight, gain=nn.init.calculate_gain("relu")
                )
                if not isinstance(layer.bias, type(None)):
                    nn.init.constant_(layer.bias, 0)
