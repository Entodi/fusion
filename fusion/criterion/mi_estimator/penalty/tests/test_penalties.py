from fusion.criterion.mi_estimator.penalty import L2Penalty
import torch
import unittest


class TestPenalties(unittest.TestCase):
    @staticmethod
    def _generate_data():
        batch_size = 2
        torch.manual_seed(42)
        x = torch.rand(batch_size, batch_size)
        return x

    def test_l2_penalty(self):
        scores = self._generate_data()
        penalty = L2Penalty()
        p = penalty(scores).item()
        self.assertAlmostEqual(p, 0.0268, places=4)


if __name__ == "__main__":
    unittest.main()
