import torch
import unittest

from fusion.architecture.dcgan import DcganDecoder


class TestDcganDecoder(unittest.TestCase):
    def test_forward(self):
        # define parameters
        input_size = 64
        dim_in = 1
        dim_h = 2
        dim_l = 4
        batch_size = 2
        # create encoder
        encoder = DcganDecoder(dim_in, dim_h, dim_l, input_size=input_size)
        # create input
        z = torch.rand(batch_size, dim_l)
        # forward pass
        output = encoder(z)
        self.assertEqual(len(output), 2)
        x, latents = output
        # check outputs
        self.assertEqual(x.size(0), batch_size)
        self.assertEqual(x.size(1), dim_in)
        self.assertEqual(x.size(2), input_size)
        self.assertEqual(x.size(3), input_size)
        self.assertEqual(latents, None)


if __name__ == "__main__":
    unittest.main()
