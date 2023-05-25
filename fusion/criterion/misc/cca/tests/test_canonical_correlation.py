from fusion.criterion.misc.cca import CanonicalCorrelation
from fusion.model.misc import ModelOutput
import torch
import unittest


class TestCanonicalCorrelation(unittest.TestCase):
    @staticmethod
    def _setup():
        seed = 42
        torch.manual_seed(seed)
        batch_size = 64
        dim_l = 64
        preds = ModelOutput(z={}, attrs={})
        sources = [0, 1]
        for source_id in sources:
            z = torch.rand(batch_size, dim_l)
            preds.z[source_id] = z
        return preds

    def test_canonincal_correlation(self):
        preds = self._setup()
        cca_loss = CanonicalCorrelation()
        loss, raw_losses = cca_loss(preds)
        keys = list(raw_losses.keys())
        self.assertAlmostEqual(loss.item(), -14.8532, places=3)
        self.assertAlmostEqual(raw_losses[keys[0]], -7.4266, places=3)
        self.assertAlmostEqual(raw_losses[keys[1]], -7.4266, places=3)


if __name__ == "__main__":
    unittest.main()
