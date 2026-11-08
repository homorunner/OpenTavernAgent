import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from agent_expand_depth import expand
from agent_depth_experiment import training_args
from agent_remote_eval import jobs_for

class ExpansionTests(unittest.TestCase):
    def test_preserves_existing_tensors_and_adds_trainable_identity_branches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source.npz'
            weights={'meta':np.frombuffer(json.dumps({'version':10,'blocks':1}).encode(),dtype=np.uint8),
                     'W1_up':np.ones((4,6),np.float32),'b1_up':np.zeros(6,np.float32),
                     'W1_down':np.ones((6,4),np.float32),'b1_down':np.zeros(4,np.float32),
                     'g1':np.ones(4,np.float32),'k1':np.zeros(4,np.float32),
                     'Wp':np.arange(12,dtype=np.float32).reshape(4,3)}
            np.savez(source,**weights)
            for depth in (1,2,4): expand(source,root/f'd{depth}.npz',depth)
            self.assertEqual(source.read_bytes(),(root/'d1.npz').read_bytes())
            with np.load(root/'d4.npz') as wide,np.load(root/'d2.npz') as short:
                self.assertEqual(json.loads(wide['meta'].tobytes())['blocks'],4)
                for k,v in weights.items():
                    if k!='meta':np.testing.assert_array_equal(v,wide[k])
                for i in (2,3,4):
                    self.assertTrue(np.any(wide[f'W{i}_up']))
                    for name in [f'W{i}_down',f'b{i}_up',f'b{i}_down',f'k{i}']:
                        self.assertFalse(np.any(wide[name]))
                    self.assertTrue(np.all(wide[f'g{i}']==1))
                np.testing.assert_array_equal(short['W2_up'],wide['W2_up'])
            with self.assertRaises(ValueError):expand(source,source,4)
            with self.assertRaises(ValueError):expand(source,root/'bad.npz',0)

    def test_starting_weights_and_step_zero_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);(work/'initial.npz').touch()
            args=training_args('agent',work,8,100,12,128)
            self.assertEqual(args[args.index('--resume')+1],str(work/'initial.npz'))
        index={'manifest':{'initial_weights':{'6':{}}},
               'checkpoints':{'d6/model.step0.ema.npz':'source'}}
        jobs=jobs_for(index)
        self.assertEqual(len(jobs),2)
        self.assertEqual({j['opponent'] for j in jobs},{'scripted','teacher'})
        self.assertTrue(all(j['step']==0 and j['args'][j['args'].index('--games')+1]=='2048' for j in jobs))

if __name__=='__main__':unittest.main()
