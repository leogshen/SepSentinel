"""Synthetic CPU-only checks; never load patient data or train models."""
import unittest
import numpy as np
from sepsis_mission_metrics import record_statistics,metrics,bootstrap

class MetricsTests(unittest.TestCase):
    def fixture(self):
        e=dict(t=np.arange(1,19),y=np.r_[np.zeros(12),np.ones(6)],eligible=np.ones(18,bool),
               onset=19,gates={'all':np.ones(18,bool),'none':np.zeros(18,bool)})
        return e,np.c_[np.zeros(18,int),np.arange(18)],np.ones(18)
    def test_alarm_matching_and_units(self):
        e,rows,p=self.fixture();m=metrics(record_statistics([e],rows,p,.5,'all'))
        self.assertEqual(m['counts']['alarms'],3);self.assertEqual(m['counts']['too_early'],2)
        self.assertEqual(m['precision'],1/3);self.assertEqual(m['recall'],1)
        self.assertEqual(m['patient_precision'],1);self.assertEqual(m['hourly_recall'],1)
    def test_duplicate_and_gate(self):
        e,rows,p=self.fixture();e['y'][6:]=1
        m=metrics(record_statistics([e],rows,p,.5,'all'))
        self.assertEqual(m['counts']['duplicate'],1)
        m=metrics(record_statistics([e],rows,p,.5,'none'))
        self.assertEqual(m['recall'],0);self.assertIsNone(m['subgroup_recall']);self.assertIsNone(m['precision'])
    def test_paired_bootstrap_identity(self):
        e,rows,p=self.fixture();s=record_statistics([e],rows,p,.5,'all')
        for v in bootstrap(s,n=20,reference=s).values():
            if v is not None:self.assertEqual(v,[0.,0.])
    def test_order_and_duplicates_rejected(self):
        e,rows,p=self.fixture()
        with self.assertRaises(ValueError):record_statistics([e],rows[::-1],p,.5,'all')
        with self.assertRaises(ValueError):record_statistics([e],np.r_[rows,rows[:1]],np.r_[p,1],.5,'all')

if __name__=='__main__':unittest.main()
