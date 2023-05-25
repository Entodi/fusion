import os
import torch
import unittest

from fusion.architecture.projection_head import LatentHead


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class TestLatentHead(unittest.TestCase):
    def test_forward(self):
        dim_in = 32
        dim_l = 64
        latent_head = LatentHead(dim_in, dim_l, use_linear=True)
        x = torch.rand((4, dim_in))
        y = latent_head.forward(x)
        self.assertEqual(y.size()[1], dim_l)


if __name__ == "__main__":
    unittest.main()
