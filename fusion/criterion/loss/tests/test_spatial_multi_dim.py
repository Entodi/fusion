import torch
import unittest

from fusion.model.dim import Dim
from fusion.utils import Setting
from fusion.criterion.loss import SpatialMultiDim
from fusion.criterion.loss.dim import CR_MODE, XX_MODE, RR_MODE, CC_MODE


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

    def test_spatial_multi_dim(self):
        output, dim_cls, estimator_setting = self._generate_output()
        criterion = SpatialMultiDim(
            dim_cls=dim_cls,
            estimator_setting=estimator_setting,
            modes=[CR_MODE, XX_MODE, CC_MODE, RR_MODE],
            weights=[1.0, 1.0, 1.0, 1.0],
        )
        total_loss, raw_losses = criterion(output)
        losses = [
            6.1393,
            0.0019,
            9.5305,
            0.5373,  # CR
            6.1271,
            0.0017,
            11.8385,
            1.0670,  # XX
            11.5989,
            1.8206,
            12.1579,
            2.2010,  # CC
            2.0716,
            0.0006,
            2.07499,
            0.0006,  # RR
        ]
        self.assertAlmostEqual(total_loss.item(), 67.1699, places=3)
        for i, (_, loss) in enumerate(raw_losses.items()):
            self.assertAlmostEqual(loss, losses[i], places=3)


if __name__ == "__main__":
    unittest.main()
