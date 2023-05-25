import os
import torch
import unittest

from fusion.architecture.projection_head import ConvHead


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class TestConvHead(unittest.TestCase):
    def test_forward(self):
        dim_in = 32
        dim_l = 64
        dim_h = 48
        conv_head = ConvHead(dim_in, dim_l, dim_h)
        x = torch.rand((4, dim_in, 32, 32))
        y = conv_head.forward(x)
        self.assertEqual(y.size()[1], dim_l)


if __name__ == "__main__":
    unittest.main()
