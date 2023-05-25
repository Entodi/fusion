import nibabel as nib
from numpy import float64

from fusion.dataset.oasis.oasis import Oasis
from fusion.dataset.misc import SetId
import unittest


class TestOasis(unittest.TestCase):
    #@unittest.skip("Skipping Oasis, as it requires OASIS which is not open sourced")
    def test_oasis(self):
        BATCH_SIZE = 8
        dataset_dir = "../../../../data/oasis_old/"
        mask = "../../../../data/MNI152_T1_3mm_brain_mask_dil_cubic192.nii.gz"
        dataset = Oasis(
            dataset_dir=dataset_dir,
            mask=mask,
            sources=[0, 1],
            batch_size=BATCH_SIZE,
            only_labeled=True,
            transforms={
                'pad_crop': True,
                'flip': False,
                'rescale': False,
                'histogram': False,
                'z_normalization': False,
                'mask': True
            },
            drop_last=True
        )
        dataset.load()
        self.assertEqual(dataset.num_classes, 2)
        self.assertEqual(len(dataset.get_loader(SetId.TRAIN)), 321)
        self.assertEqual(len(dataset.get_loader(SetId.VALID)), 81)
        self.assertEqual(len(dataset.get_loader(SetId.INFER)), 11)
        self.assertEqual(len(dataset.get_cv_loaders()), 2)
        self.assertEqual(len(dataset.get_all_loaders()), 3)

if __name__ == "__main__":
    unittest.main()
