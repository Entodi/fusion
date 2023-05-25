import torch
import unittest

from fusion.architecture.dcgan import DcganAutoEncoder


class TestDcganAutoEncoder(unittest.TestCase):
    def test_forward(self):
        # define parameters
        input_size = 32
        dim_in = 1
        dim_h = 2
        dim_l = 4
        dim_cls = [8]
        batch_size = 2
        # create encoder
        encoder = DcganAutoEncoder(dim_in, dim_h, dim_l, dim_cls, input_size=input_size)
        # create input
        x = torch.rand(batch_size, dim_in, input_size, input_size)
        # forward pass
        output = encoder(x)
        # check outputs
        self.assertEqual(len(output), 2)
        z, x_hat = output
        self.assertEqual(z.size(0), batch_size)
        self.assertEqual(z.size(1), dim_l)
        self.assertEqual(x_hat.size(), x.size())


if __name__ == "__main__":
    unittest.main()
