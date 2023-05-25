import numpy as np

from fusion.architecture.dcgan import DcganEncoder
from fusion.dataset.misc import SetId
from fusion.dataset.ukbiobank import UKBioBank, UKBioBankMP, UKBioBankRAM, UKBioBankMPRAM

import torch
import unittest


class TestUKBioBank(unittest.TestCase):
    def test_ukbiobank_ram(self):
        expected_shape = [1, 1, 128, 128, 128]
        batch_size = 1
        sources = range(3)
        dataset_dir = "../../../../data/ukbiobank/folds/"
        dataset = UkBioBankRAM(
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            sources=sources
        )
        dataset.load()
        self.assertEqual(dataset.num_classes, 10)
        self.assertEqual(len(dataset.get_loader(SetId.TRAIN)), 1600)
        self.assertEqual(len(dataset.get_loader(SetId.VALID)), 200)
        self.assertEqual(len(dataset.get_loader(SetId.INFER)), 200)
        self.assertEqual(len(dataset.get_cv_loaders()), 2)
        self.assertEqual(len(dataset.get_all_loaders()), 3)
        for sample in dataset.get_loader(SetId.VALID):
            for sourse_id in sources:
                print (sample[f'source_{sourse_id}']['data'].max())
                print (sample[f'source_{sourse_id}']['data'].min())
                shape = sample[f'source_{sourse_id}']['data'].shape
                print (shape)
                self.assertTrue(
                    np.array_equal(shape, expected_shape, equal_nan=True))
            break

    def test_ukbiobank(self):
        expected_shape = [1, 1, 128, 128, 128]
        batch_size = 1
        sources = range(3)
        dataset_dir = "../../../../data/ukbiobank/folds/"
        dataset = UKBioBank(
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            sources=sources
        )
        dataset.load()
        self.assertEqual(dataset.num_classes, 10)
        self.assertEqual(len(dataset.get_loader(SetId.TRAIN)), 1600)
        self.assertEqual(len(dataset.get_loader(SetId.VALID)), 200)
        self.assertEqual(len(dataset.get_loader(SetId.INFER)), 200)
        self.assertEqual(len(dataset.get_cv_loaders()), 2)
        self.assertEqual(len(dataset.get_all_loaders()), 3)
        for sample in dataset.get_loader(SetId.VALID):
            for sourse_id in sources:
                print (sample[f'source_{sourse_id}']['data'].max())
                print (sample[f'source_{sourse_id}']['data'].min())
                shape = sample[f'source_{sourse_id}']['data'].shape
                print (shape)
                self.assertTrue(
                    np.array_equal(shape, expected_shape, equal_nan=True))
            break

    def test_ukbiobank_mp(self):
        expected_shape = [1, 1, 128, 128, 128]
        batch_size = 1
        sourse_id = 0
        dataset_dir = "../../../../data/ukbiobank/folds/"
        dataset = UKBioBankMP(
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            shuffle=True,
        )
        dataset.load()
        self.assertEqual(dataset.num_classes, 10)
        self.assertEqual(len(dataset.get_loader(SetId.TRAIN)), 4800)
        self.assertEqual(len(dataset.get_loader(SetId.VALID)), 600)
        self.assertEqual(len(dataset.get_loader(SetId.INFER)), 600)
        self.assertEqual(len(dataset.get_cv_loaders()), 2)
        self.assertEqual(len(dataset.get_all_loaders()), 3)
        for sample in dataset.get_loader(SetId.VALID):
            print (sample[f'source_{sourse_id}']['data'].max())
            print (sample[f'source_{sourse_id}']['data'].min())
            shape = sample[f'source_{sourse_id}']['data'].shape
            print (sample[f'pipeline_{sourse_id}'])
            print (shape)
            self.assertTrue(
                np.array_equal(shape, expected_shape, equal_nan=True))
            break

    def test_ukbiobank_mp_ram(self):
        expected_shape = [1, 1, 128, 128, 128]
        batch_size = 1
        sourse_id = 0
        dataset_dir = "../../../../data/ukbiobank/folds/"
        dataset = UKBioBankMPRAM(
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            shuffle=True,
        )
        dataset.load()
        self.assertEqual(dataset.num_classes, 10)
        self.assertEqual(len(dataset.get_loader(SetId.TRAIN)), 4800)
        self.assertEqual(len(dataset.get_loader(SetId.VALID)), 600)
        self.assertEqual(len(dataset.get_loader(SetId.INFER)), 600)
        self.assertEqual(len(dataset.get_cv_loaders()), 2)
        self.assertEqual(len(dataset.get_all_loaders()), 3)
        for sample in dataset.get_loader(SetId.VALID):
            print (sample[f'source_{sourse_id}']['data'].max())
            print (sample[f'source_{sourse_id}']['data'].min())
            shape = sample[f'source_{sourse_id}']['data'].shape
            print (sample[f'pipeline_{sourse_id}'])
            print (shape)
            self.assertTrue(
                np.array_equal(shape, expected_shape, equal_nan=True))
            break



if __name__ == "__main__":
    unittest.main()
