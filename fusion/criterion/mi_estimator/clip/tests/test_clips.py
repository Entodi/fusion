from fusion.criterion.mi_estimator.clip import TahnClip
import torch
import unittest


class TestClips(unittest.TestCase):
    @staticmethod
    def _generate_data():
        batch_size = 2
        torch.manual_seed(42)
        x = torch.rand(batch_size, batch_size)
        return x

    def test_tahn_clip(self):
        clip_value = 10.0
        scores = self._generate_data()
        clip = TahnClip(clip_value=clip_value)
        clipped_scores = clip(scores)
        self.assertAlmostEqual(clipped_scores[0, 0].item(), 0.8800, places=4)
        self.assertAlmostEqual(clipped_scores[0, 1].item(), 0.9125, places=4)
        self.assertAlmostEqual(clipped_scores[1, 0].item(), 0.3827, places=4)
        self.assertAlmostEqual(clipped_scores[1, 1].item(), 0.9564, places=4)
        score = torch.FloatTensor([100, -100])
        clipped_score = clip(score)
        self.assertEqual(clipped_score[0].item(), clip_value)
        self.assertEqual(clipped_score[1].item(), -clip_value)


if __name__ == "__main__":
    unittest.main()
