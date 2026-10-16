"""Verify that lab numeric information is removed without losing its observation pattern."""
import unittest
import numpy as np
import torch
from transfer_sequence import collate
from causal_control_features import prefix_features
from lab_value_ablation import LAB_INDICES, BLIND_COLUMNS, blind_collate, blind_table


class LabValueContract(unittest.TestCase):
    def test_table_and_neural_information(self):
        rng = np.random.default_rng(391)
        raw = rng.normal(size=(80, 34)).astype(np.float32)
        raw[rng.random(raw.shape) < .55] = np.nan
        changed = raw.copy()
        changed[:, list(LAB_INDICES)] *= -3
        e = dict(x=raw, t=np.arange(1, 81, dtype=np.float32), static=np.ones(4), kind=np.zeros(34, np.int64))
        other = dict(e, x=changed)
        med, std = np.zeros(34), np.ones(34)
        original = prefix_features(e, med, std)
        masked = blind_table(original)
        altered = blind_table(prefix_features(other, med, std))
        np.testing.assert_array_equal(masked, altered)
        keep = np.setdiff1d(np.arange(original.shape[1]), BLIND_COLUMNS)
        np.testing.assert_array_equal(masked[:, keep], original[:, keep])
        self.assertTrue(np.all(masked[:, BLIND_COLUMNS] == 0))
        full = collate([e], med, std, device='cpu')
        a = blind_collate([e], med, std, device='cpu')
        b = blind_collate([other], med, std, device='cpu')
        for first, second in zip(a, b):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(a[0][..., 1:], full[0][..., 1:], rtol=0, atol=0)
        nonlab = np.setdiff1d(np.arange(34), LAB_INDICES).tolist()
        torch.testing.assert_close(a[0][:, :, nonlab], full[0][:, :, nonlab], rtol=0, atol=0)
        self.assertTrue(torch.all(a[0][:, :, list(LAB_INDICES), 0] == 0))
        # The oxygen-support setting remains available numerically in both arms.
        self.assertNotIn(16, LAB_INDICES)
        self.assertEqual(len(LAB_INDICES), 25)


if __name__ == '__main__':
    unittest.main()
