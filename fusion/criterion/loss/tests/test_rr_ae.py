import torch
import unittest

from fusion.model.ae import AE
from fusion.utils import Setting
from fusion.criterion.loss import RR_AE


class TestSpatialMultiDim(unittest.TestCase):
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
        return output, estimator_setting

    def test_rr_ae(self):
        output, estimator_setting = self._generate_output()
        criterion = RR_AE(
            estimator_setting=estimator_setting,
        )
        total_loss, raw_losses = criterion(output)
        print(total_loss, raw_losses)
        self.assertAlmostEqual(total_loss.item(), 8.2415, places=3)
        self.assertAlmostEqual(raw_losses["AE_0"], 0.5844, places=3)
        self.assertAlmostEqual(raw_losses["AE_1"], 0.5333, places=3)
        self.assertAlmostEqual(raw_losses["RR1_0_1_loss"], 3.3395, places=3)
        self.assertAlmostEqual(raw_losses["RR1_0_1_penalty"], 0.1808, places=3)
        self.assertAlmostEqual(raw_losses["RR1_1_0_loss"], 3.4225, places=3)
        self.assertAlmostEqual(raw_losses["RR1_1_0_penalty"], 0.1808, places=3)


if __name__ == "__main__":
    unittest.main()
