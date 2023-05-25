import torch
import unittest

from fusion.model.ae import AE


class TestAE(unittest.TestCase):
    def test_forward(self):
        # define parameters
        dim_in = 1
        dim_l = 4
        input_size = 32
        architecture = "DcganAutoEncoder"
        architecture_params = dict(
            input_size=input_size, dim_in=[dim_in], dim_h=2, dim_l=dim_l
        )
        sources = [0]
        batch_size = 2
        # create model
        model = AE(sources, architecture, architecture_params)
        # create input
        x = [torch.rand(batch_size, dim_in, input_size, input_size)]
        # forward pass
        output = model(x)
        # check outputs
        for _, z in output.z.items():
            self.assertEqual(z.size(1), dim_l)
        self.assertEqual(output.attrs["x"][0].size(), x[0].size())
        self.assertEqual(output.attrs["x_hat"][0].size(), x[0].size())


if __name__ == "__main__":
    unittest.main()
