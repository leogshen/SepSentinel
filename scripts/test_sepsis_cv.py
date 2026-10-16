import unittest
import numpy as np
import run_sepsis_cv as cv

class SplitContracts(unittest.TestCase):
    def test_outer_coverage_and_inner_isolation(self):
        labels=np.r_[np.zeros(900,dtype=int),np.ones(100,dtype=int)]
        splits=list(cv.partitions(labels)); seen=[]
        for s in splits:
            groups=[set(x.tolist()) for x in s.values()]
            self.assertEqual(len(set.union(*groups)),1000)
            for i in range(4):
                for j in range(i): self.assertFalse(groups[i]&groups[j])
            self.assertEqual(int(labels[s['test']].sum()),20)
            seen.extend(s['test'].tolist())
        self.assertEqual(sorted(seen),list(range(1000)))
        repeat=list(cv.partitions(labels))
        for a,b in zip(splits,repeat):
            for k in a: np.testing.assert_array_equal(a[k],b[k])

    def test_scaling_uses_only_fit_records(self):
        fit=[dict(x=np.ones((8,34),np.float32)),dict(x=np.full((8,34),3,np.float32))]
        heldout=dict(x=np.full((8,34),100,np.float32))
        first=cv.normalization(fit)
        heldout['x'][:]=1e9
        second=cv.normalization(fit)
        np.testing.assert_array_equal(first['median'],second['median'])
        np.testing.assert_allclose(first['median'][10:],2.)
        changed=cv.normalization(fit+[heldout])
        self.assertFalse(np.array_equal(first['median'][10:],changed['median'][10:]))

if __name__=='__main__': unittest.main()
