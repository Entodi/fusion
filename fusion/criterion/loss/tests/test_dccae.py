import torch
import unittest

from fusion.model.ae import AE
from fusion.criterion.loss import DCCAE


class TestDCCAE(unittest.TestCase):
    @staticmethod
    def _generate_output():
        torch.manual_seed(42)
        dim_in = 1
        dim_l = 64
        input_size = 32
        architecture = "DcganAutoEncoder"
        architecture_params = dict(
            input_size=input_size,
            dim_in=[dim_in, dim_in],
            dim_h=2,
            dim_l=dim_l,
            dim_cls=[],
        )
        sources = [0, 1]
        batch_size = 8
        # create model
        model = AE(sources, architecture, architecture_params)
        # create input
        x = []
        for _ in sources:
            x.append(torch.rand(batch_size, dim_in, input_size, input_size))
        # forward pass
        output = model(x)
        return output

    def test_dccae(self):
        output = self._generate_output()
        criterion = DCCAE()
        total_loss, raw_losses = criterion(output)
        self.assertAlmostEqual(total_loss.item(), -4.1737, places=3)
        self.assertAlmostEqual(raw_losses["AE_0"], 0.5844, places=3)
        self.assertAlmostEqual(raw_losses["AE_1"], 0.5333, places=3)
        self.assertAlmostEqual(raw_losses["CCA_0_1"], -2.6457, places=3)
        self.assertAlmostEqual(raw_losses["CCA_1_0"], -2.6457, places=3)


if __name__ == "__main__":
    unittest.main()
