from fusion.criterion.loss.dim import CrDim, CcDim, RrDim, XxDim
from fusion.criterion.mi_estimator import InfoNceEstimator
from fusion.utils import Setting
import torch
import unittest


class TestDim(unittest.TestCase):
    @staticmethod
    def _setup():
        torch.manual_seed(42)
        sources = [0, 1]
        conv_latent_size = [32, 1]
        batch_size = 8
        dim_l = 64
        convs = {}
        reps = {}
        for source_id in sources:
            reps[source_id] = {}
            convs[source_id] = {}
            for dim_conv in conv_latent_size:
                locations = dim_conv
                data = torch.rand(batch_size, dim_l, locations)
                if dim_conv == 1:
                    reps[source_id][dim_conv] = data
                else:
                    reps[source_id][dim_conv] = torch.rand(batch_size, dim_l, 1)
                    convs[source_id][dim_conv] = data

        critic_setting = Setting(class_type="SeparableCritic", args={})
        clip_setting = Setting(class_type="TahnClip", args={})
        penalty_setting = Setting(class_type="L2Penalty", args={})
        estimator = InfoNceEstimator(
            critic_setting, clip_setting, penalty_setting=penalty_setting
        )

        return convs, reps, estimator

    def test_cr_dim(self):
        convs, reps, estimator = self._setup()
        objective = CrDim(estimator=estimator, weight=1)
        loss, raw_losses = objective(reps, convs)
        raw_keys = list(raw_losses.keys())
        self.assertAlmostEqual(raw_losses[raw_keys[0]], 5.4389, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[1]], 10.7434, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[2]], 5.4361, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[3]], 10.4492, places=3)
        self.assertAlmostEqual(loss.item(), 32.0676, places=3)

    def test_cc_dim(self):
        convs, reps, estimator = self._setup()
        objective = CcDim(estimator=estimator, weight=1)
        loss, raw_losses = objective(reps, convs)
        raw_keys = list(raw_losses.keys())
        self.assertAlmostEqual(raw_losses[raw_keys[0]], 5.4396, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[1]], 10.2893, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[2]], 5.4479, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[3]], 10.3638, places=3)
        self.assertAlmostEqual(loss.item(), 31.5406, places=3)

    def test_xx_dim(self):
        convs, reps, estimator = self._setup()
        objective = XxDim(estimator=estimator, weight=1)
        loss, raw_losses = objective(reps, convs)
        raw_keys = list(raw_losses.keys())
        self.assertAlmostEqual(raw_losses[raw_keys[0]], 5.4481, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[1]], 10.7993, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[2]], 5.4465, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[3]], 10.3911, places=3)
        self.assertAlmostEqual(loss.item(), 32.0850, places=3)

    def test_rr_dim(self):
        convs, reps, estimator = self._setup()
        objective = RrDim(estimator=estimator, weight=1)
        loss, raw_losses = objective(reps, convs)
        raw_keys = list(raw_losses.keys())
        print(loss, raw_losses)
        self.assertAlmostEqual(raw_losses[raw_keys[0]], 2.0777, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[1]], 10.9743, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[2]], 2.0887, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[3]], 10.9743, places=3)
        self.assertAlmostEqual(loss.item(), 26.1152, places=3)


if __name__ == "__main__":
    unittest.main()
