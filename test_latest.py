import unittest
import numpy as np
from scheduler_comparison_final import problem
from run_latest import fim_candidates,augment
from candidate_planning import GeometryFactors,cost
from evidence_certificates import PlanCheck

class R30Tests(unittest.TestCase):
 def test_fim_family_preserves_catalog_and_masks(self):
  p=problem(7999901,4,5,.1);o,reg,cap,valid=p[3],p[5],p[7],p[11]
  paths=fim_candidates(o,reg,cap,valid)
  self.assertTrue(paths)
  for path in paths:
   self.assertEqual(len(path),5)
   self.assertTrue(all(valid[t][a] for t,a in enumerate(path)))
  self.assertEqual(len(reg.keys),30)
  self.assertEqual(fim_candidates(o,reg,cap,[np.zeros_like(v,bool) for v in valid]),[])
 def test_preserves_candidate_and_original_checker(self):
  p=problem(7999902,4,5,.1);o,reg,cap,valid=p[3],p[5],p[7],p[11];f=GeometryFactors(o,reg)
  rich=tuple(len(g)-1 for g in o.catalogs)
  self.assertLess(f.check(rich,cap),.1)
  out,n=augment(f,cap,valid,[rich]*3)
  for path in out:
   self.assertLessEqual(cost(o,path),cost(o,rich)+1e-10)
   self.assertAlmostEqual(f.check(path,cap),PlanCheck(o,reg,path,6).evaluate(cap),places=10)
 def test_crossover_units(self):
  reserve,fb,A,L,J=.002685,.001598,.8,2,20
  h=(reserve-fb+A/L)/J
  self.assertAlmostEqual(reserve+A/L,fb+J*h)
  self.assertGreater(h,(reserve-fb)/J)

if __name__=='__main__':unittest.main()
