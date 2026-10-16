import unittest
import numpy as np
import torch
import run_lab_specialization as job

class Contracts(unittest.TestCase):
    def test_gate_and_history_contract_on_real_fold(self):
        job.configure(0)
        eps=job.RAW_LOAD('fit')
        for e in eps:
            x,t=e['x'],e['t']
            last=np.maximum.accumulate(np.where(np.isfinite(x),t[:,None],-np.inf),axis=0)
            age=t[:,None]-last
            gate=(age[:,10:13]<2).any(1)&(age[:,[6,7,8]]<24).all(1)&(age[:,[4,5]]<24).any(1)
            np.testing.assert_array_equal(gate,e['gates'][job.GATE])
        rich=job.restricted(eps)
        self.assertGreater(sum(e['eligible'].sum() for e in rich),0)
        for a,b in zip(eps,rich):
            self.assertIs(a['x'],b['x'])
            np.testing.assert_array_equal(b['eligible'],a['eligible']&a['gates'][job.GATE])
            np.testing.assert_array_equal(a['y'],b['y'])
        # Evaluation retains the broad event denominator; eligibility is not gated.
        evaluation=job.load('cal'); raw=job.RAW_LOAD('cal')
        self.assertTrue(any((e['eligible']&~e['gates'][job.GATE]).any() for e in evaluation))
        for a,b in zip(evaluation,raw): np.testing.assert_array_equal(a['eligible'],b['eligible'])

    def test_budgets_and_matched_heads(self):
        self.assertFalse(job.stop(29,40)); self.assertTrue(job.stop(30,30))
        self.assertFalse(job.stop(100,29)); self.assertTrue(job.stop(200,0))
        arms=job.candidate_arms(); self.assertEqual(len(arms),9)
        for method in job.METHODS:
            self.assertEqual([a['lr'] for a in arms if a['method']==method],job.C['learning_rates'])
        torch.manual_seed(42); a=job.control.base.RichModel(34)
        torch.manual_seed(42); b=job.control.base.RichModel(34)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.head.parameters(),b.head.parameters())))

if __name__=='__main__': unittest.main()
