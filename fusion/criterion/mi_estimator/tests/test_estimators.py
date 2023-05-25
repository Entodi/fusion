from fusion.criterion.mi_estimator import InfoNceEstimator
from fusion.criterion.mi_estimator import FenchelDualEstimator
from fusion.utils import Setting
import torch
import unittest


class TestMIEstimators(unittest.TestCase):
    @staticmethod
    def _generate_data():
        batch_size = 8
        dim_l = 64
        locs = 32
        zs = 1
        torch.manual_seed(42)
        x = torch.rand(batch_size, dim_l, locs)
        y = torch.rand(batch_size, dim_l, zs)
        return x, y

    def test_infonce_estimator(self):
        critic_setting = Setting(class_type="SeparableCritic", args={})
        clip_setting = Setting(class_type="TahnClip", args={})
        penalty_setting = Setting(class_type="L2Penalty", args={})
        estimator = InfoNceEstimator(
            critic_setting, clip_setting, penalty_setting=penalty_setting
        )
        x, y = self._generate_data()
        score, penalty = estimator(x, y)
        score = score.item()
        penalty = penalty.item()
        self.assertAlmostEqual(score, 5.4309, places=3)
        self.assertAlmostEqual(penalty, 10.2132, places=3)

    def test_fenchel_dual(self):
        critic_setting = Setting(class_type="SeparableCritic", args={})
        penalty_setting = Setting(class_type="L2Penalty", args={})
        estimator = FenchelDualEstimator(
            critic_setting, penalty_setting=penalty_setting
        )
        x, y = self._generate_data()
        score, penalty = estimator(x, y)
        score = score.item()
        penalty = penalty.item()
        self.assertAlmostEqual(score, 14.4711, places=3)
        self.assertAlmostEqual(penalty, 10.2132, places=3)


if __name__ == "__main__":
    unittest.main()
