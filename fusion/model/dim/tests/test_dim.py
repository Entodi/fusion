from fusion.model.dim import Dim
import torch
import unittest


class TestDim(unittest.TestCase):
    def test_forward(self):
        # define parameters
        dim_in = 1
        dim_l = 4
        dim_cls = [8]
        input_size = 32
        architecture = "DcganEncoder"
        architecture_params = dict(
            input_size=input_size,
            dim_in=[dim_in, dim_in],
            dim_h=2,
            dim_l=dim_l,
            dim_cls=dim_cls,
        )
        sources = [0, 1]
        batch_size = 2
        # create model
        model = Dim(sources, architecture, architecture_params)
        # create input
        x = []
        for _ in sources:
            x.append(torch.rand(batch_size, dim_in, input_size, input_size))
        # forward pass
        output = model(x)
        # check outputs
        for _, latent in output.z.items():
            self.assertEqual(latent.size(1), dim_l)
        for source_id in sources:
            for dim_conv in dim_cls:
                self.assertEqual(
                    output.attrs["latents"][source_id][dim_conv].size(-1), dim_conv
                )


if __name__ == "__main__":
    unittest.main()
