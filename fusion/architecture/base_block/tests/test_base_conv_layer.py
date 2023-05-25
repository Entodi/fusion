import torch
import torch.nn as nn
import unittest

from fusion.architecture.base_block import BaseConvLayer


class TestBaseConvLayer(unittest.TestCase):
    def test_forward(self):
        base_layer = BaseConvLayer(
            conv_layer_class=nn.Conv1d,
            conv_layer_args={
                "in_channels": 1,
                "out_channels": 1,
                "kernel_size": 1,
            },
        )
        self.assertEqual(len(base_layer.forward(torch.rand(2, 1, 2))), 2)


if __name__ == "__main__":
    unittest.main()
