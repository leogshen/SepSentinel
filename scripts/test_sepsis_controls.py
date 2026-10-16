"""Independent causal-feature and pretrained-weight contract checks; no clinical fitting."""
import unittest
import numpy as np
import torch
from causal_control_features import prefix_features, columns, WINDOWS, STATS
from transfer_sequence import collate
import train_rich_transfer as base


class ControlContracts(unittest.TestCase):
    def episode(self):
        rng = np.random.default_rng(91)
        x = rng.normal(size=(90, 34)).astype(np.float32)
        x[rng.random(x.shape) < .6] = np.nan
        return dict(x=x, t=np.arange(1, 91, dtype=np.float32), static=np.array([.5, 1, 1, 1]),
                    kind=np.zeros(34, np.int64))

    def test_future_and_prefix_invariance(self):
        e = self.episode()
        med, std = np.zeros(34), np.ones(34)
        full = prefix_features(e, med, std)
        for j in [0, 5, 11, 23, 71, 89]:
            short = dict(e, x=e['x'][:j+1], t=e['t'][:j+1])
            np.testing.assert_allclose(full[j], prefix_features(short, med, std)[-1], atol=1e-6)
        changed = dict(e, x=e['x'].copy())
        changed['x'][40:] = 1e5
        np.testing.assert_array_equal(full[:40], prefix_features(changed, med, std)[:40])

    def test_summary_against_direct_observations(self):
        e = self.episode()
        z = prefix_features(e, np.zeros(34), np.ones(34))
        for j in [2, 8, 27, 77, 89]:
            for wi, width in enumerate(WINDOWS):
                for f in [0, 9, 33]:
                    start = max(0, j-width+1)
                    v = e['x'][start:j+1, f]
                    t = e['t'][start:j+1]
                    valid = np.isfinite(v)
                    got = z[j, wi*340+f*10:wi*340+f*10+10]
                    if not valid.any():
                        np.testing.assert_array_equal(got, [0]*8+[width, 0])
                    else:
                        v, t = v[valid], t[valid]
                        slope = np.polyfit(t, v, 1)[0] if len(v) > 1 else 0
                        expected = [v[-1], v.mean(), v.std(), v.min(), v.max(), slope,
                                    v[-1]-v[0], len(v), e['t'][j]-t[-1], 1]
                        np.testing.assert_allclose(got, expected, atol=2e-6)
        self.assertEqual(len(columns(10)), 405)
        self.assertEqual(len(columns(34)), z.shape[1])

    def test_mask_excludes_inactive_features(self):
        e = self.episode()
        other = dict(e, x=e['x'].copy())
        other['x'][:, 10:] = 1000
        a = prefix_features(e, np.zeros(34), np.ones(34))[:, columns(10)]
        b = prefix_features(other, np.zeros(34), np.ones(34))[:, columns(10)]
        np.testing.assert_array_equal(a, b)

    def test_source_load_and_causal_encoder(self):
        torch.manual_seed(42)
        scratch = base.RichModel(34).eval()
        torch.manual_seed(42)
        source = base.RichModel(34).eval()
        source.source_init(42)
        original = torch.load(base.OLD/'mae_seed42.pt', map_location='cpu', weights_only=False)['encoder']
        for key, value in source.encoder.state_dict().items():
            expected = original[key]
            actual = value[:10] if key == 'feature_id.weight' else value
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for a, b in zip(source.head.parameters(), scratch.head.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(source.encoder.feature_id.weight[10:], scratch.encoder.feature_id.weight[10:])
        e = self.episode()
        short = dict(e, x=e['x'][:12], t=e['t'][:12])
        norm = (np.zeros(34), np.ones(34))
        with torch.no_grad():
            full = source.encoder(*collate([e], *norm, device='cpu'))[0, :12]
            prefix = source.encoder(*collate([short], *norm, device='cpu'))[0]
            padded = source.encoder(*collate([short, e], *norm, device='cpu'))[0, :12]
        torch.testing.assert_close(full, prefix, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(padded, prefix, rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
