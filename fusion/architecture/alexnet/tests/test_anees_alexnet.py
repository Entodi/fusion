import torch
import unittest

from fusion.architecture.alexnet import AneesAlexNet


class TestAlexNetEncoder(unittest.TestCase):
    def test_forward(self):
        input_size = 128
        dim_in = 1
        dim_l = 4
        batch_size = 2
        encoder = AneesAlexNet(dim_in, dim_l)
        x = torch.rand(batch_size, dim_in, input_size, input_size)
        output = encoder(x)
        z, _  = output
        print (z.size())
        self.assertEqual(z.size(0), batch_size)
        self.assertEqual(z.size(1), dim_l)


if __name__ == "__main__":
    unittest.main()
