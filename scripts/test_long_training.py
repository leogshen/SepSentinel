"""Stopping-rule and CUDA recovery tests for the longer experiment."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from run_long_sepsis_training import stop_training, save_checkpoint, arms, control


class LongTrainingTests(unittest.TestCase):
    def test_stopping_budget(self):
        c = dict(epochs=100, min_epochs=20, patience=15)
        self.assertFalse(stop_training(19, 18, c))
        self.assertTrue(stop_training(20, 18, c))
        self.assertFalse(stop_training(35, 14, c))
        self.assertTrue(stop_training(36, 15, c))
        self.assertTrue(stop_training(100, 0, c))
        planned = arms()
        self.assertEqual(len(planned), 27)
        self.assertEqual(sum(a['condition']=='availability' for a in planned), 3)
        for a in planned:
            if a['condition']=='availability':
                self.assertTrue(any(b['name']==a['name'].removeprefix('availability__') for b in planned))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_optimizer_and_rng_recovery(self):
        torch.manual_seed(28)
        rng = np.random.default_rng(28)
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.2),
                                    torch.nn.Linear(8, 1)).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.0003)
        x = torch.arange(16, dtype=torch.float32, device='cuda').reshape(4, 4)/16

        def step():
            model.train()
            opt.zero_grad(set_to_none=True)
            ix = rng.permutation(4)
            model(x[ix]).square().mean().backward()
            opt.step()

        step()
        with tempfile.TemporaryDirectory(dir=control.ROOT.parent) as directory:
            path = Path(directory)/'recovery.pt'
            save_checkpoint(path, dict(model=model.state_dict(), optimizer=opt.state_dict(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                sample_rng=rng.bit_generator.state))
            step()
            expected = {k:v.detach().clone() for k,v in model.state_dict().items()}
            saved = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(saved['model'])
            opt.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['torch_rng'])
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
            rng.bit_generator.state = saved['sample_rng']
            step()
            for k,v in model.state_dict().items():
                torch.testing.assert_close(v, expected[k], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
