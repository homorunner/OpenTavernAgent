import unittest
from collections import Counter
from pathlib import Path
import tempfile

from agent_league_sampling import weighted_tables
from agent_remote_eval import league_jobs


class SamplingTests(unittest.TestCase):
    def test_new_models_are_prioritized_and_old_models_still_play(self):
        initial={**{f'new{i}':0 for i in range(8)},**{f'old{i}':1000000 for i in range(24)}}
        counts=dict(initial)
        tables=weighted_tables(counts,512,42)
        appearances=Counter(n for t in tables for n in t)
        self.assertTrue(all(len(t)==len(set(t))==8 for t in tables))
        self.assertGreater(sum(appearances[f'new{i}'] for i in range(8))/8,
                           4*sum(appearances[f'old{i}'] for i in range(24))/24)
        self.assertTrue(all(appearances[f'old{i}']>0 for i in range(24)))
        self.assertEqual(counts,{n:initial[n]+appearances[n] for n in initial})
        self.assertEqual(tables,weighted_tables(dict(initial),512,42))

    def test_planned_and_running_tables_reserve_counts(self):
        index={'checkpoints':{f'd{d}/model.step100{s}':'path' for d in [6,8,10,12] for s in ['.npz','.ema.npz']}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            jobs=league_jobs(index,root,{},2)
            self.assertTrue(all(v==0 for v in jobs[0]['sampling']['counts_before'].values()))
            self.assertTrue(all(v==128 for v in jobs[1]['sampling']['counts_before'].values()))
            next_jobs=league_jobs(index,root,{'active':(jobs[0],{})},1)
            self.assertEqual(next_jobs[0]['sampling']['counts_before'],jobs[1]['sampling']['counts_before'])
            self.assertEqual(next_jobs[0]['tables'],jobs[1]['tables'])


if __name__=='__main__': unittest.main()
