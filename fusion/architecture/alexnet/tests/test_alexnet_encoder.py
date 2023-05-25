import torch
import unittest

from fusion.architecture.alexnet import AlexNetEncoder


class TestAlexNetEncoder(unittest.TestCase):
    def test_forward(self):
        input_size = 128
        dim_in = 1
        dim_h = 2
        dim_l = 4
        dim_cls = [6]
        batch_size = 2
        encoder = AlexNetEncoder(dim_in, dim_l, dim_cls=dim_cls, input_size=input_size)
        x = torch.rand(batch_size, dim_in, input_size, input_size)
        output = encoder(x)
        z, latents = output
        print (z.size())
        self.assertEqual(z.size(0), batch_size)
        self.assertEqual(z.size(1), dim_l)
        for i, (d, l) in enumerate(latents.items()):
            if d != 1:
                self.assertEqual(l.size(0), batch_size)
                self.assertEqual(l.size(-1), dim_cls[i])
                self.assertEqual(len(l.size()), len(x.size()))


if __name__ == "__main__":
    unittest.main()
