import torch
import unittest

from fusion.model.dim import Dim
from fusion.utils import Setting
from fusion.criterion.loss import CR_CCA


class TestSpatialMultiDim(unittest.TestCase):
    @staticmethod
    def _generate_output():
        torch.manual_seed(42)
        dim_in = 1
        dim_l = 64
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
        batch_size = 8
        # create model
        model = Dim(sources, architecture, architecture_params)
        # create input
        x = []
        for _ in sources:
            x.append(torch.rand(batch_size, dim_in, input_size, input_size))
        # forward pass
        output = model(x)
        critic_setting = Setting(class_type="SeparableCritic", args={})
        clip_setting = Setting(class_type="TahnClip", args={})
        penalty_setting = Setting(class_type="L2Penalty", args={})
        estimator_setting = Setting(
            class_type="InfoNceEstimator",
            args={
                "critic_setting": critic_setting,
                "clip_setting": clip_setting,
                "penalty_setting": penalty_setting,
            },
        )
        return output, dim_cls, estimator_setting

    def test_cr_cca(self):
        output, dim_cls, estimator_setting = self._generate_output()
        criterion = CR_CCA(
            dim_cls=dim_cls,
            input_dim=2,
            estimator_setting=estimator_setting,
            cca_args={},
        )
        total_loss, raw_losses = criterion(output)
        self.assertAlmostEqual(total_loss.item(), 10.9177, places=3)
        self.assertAlmostEqual(raw_losses["CR8_0_loss"], 6.1393, places=3)
        self.assertAlmostEqual(raw_losses["CR8_0_penalty"], 0.0019, places=3)
        self.assertAlmostEqual(raw_losses["CR8_1_loss"], 9.5305, places=3)
        self.assertAlmostEqual(raw_losses["CR8_1_penalty"], 0.5373, places=3)
        self.assertAlmostEqual(raw_losses["CCA_0_1"], -2.6457, places=3)
        self.assertAlmostEqual(raw_losses["CCA_1_0"], -2.6457, places=3)


if __name__ == "__main__":
    unittest.main()
