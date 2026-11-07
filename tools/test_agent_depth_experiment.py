"""Scheduler tests with small fake subprocesses; no game simulation or NUMA hardware."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import agent_depth_experiment as runner


class IndependentSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.args = SimpleNamespace(binary=root/'tavernagent', work=root/'work',
                                    archive=root/'archive', seed=1, workers=1)
        self.args.archive.mkdir()
        self.calls = []
        self.lock = threading.Lock()
        self.active = set()

    def fake_job(self, slot, cmd, log, workers, abort=None):
        with self.lock:
            self.assertNotIn(slot, self.active, 'Jobs overlap on one NUMA node')
            self.active.add(slot)
            self.calls.append((slot, cmd))
        try:
            out = Path(cmd[cmd.index('--out')+1])
            out.parent.mkdir(parents=True, exist_ok=True)
            if cmd[1] == 'selfplay':
                depth = int(cmd[cmd.index('--blocks')+1])
                stage = int(cmd[cmd.index('--steps')+1])
                if hasattr(self, 'barrier_test'):
                    if depth == 12 and stage == 100:
                        self.assertTrue(self.barrier_test.wait(5), 'Fast lane waited for slow lane')
                    if depth == 6 and stage == 1000:
                        self.barrier_test.set()
                state = dict(step=stage, format=8, net=dict(hidden=256, embedding_dim=32, blocks=depth))
                (out.parent/'model.trainer.npz').write_text(json.dumps(state))
                (out.parent/'model.history.json').write_text(json.dumps([
                    dict(step=i, collect_seconds=1, train_seconds=2) for i in range(1, stage+1)]))
                for suffix in ['.npz', '.ema.npz']:
                    (out.parent/f'model.step{stage}{suffix}').write_text(f'{depth}:{stage}')
                    (out.parent/f'model{suffix}').write_text(f'{depth}:{stage}')
            elif cmd[1] == 'arena':
                for flag in ['--a', '--b']:
                    self.assertTrue(Path(cmd[cmd.index(flag)+1]).is_file())
                out.write_text('{}')
            elif cmd[1] == 'eval':
                self.assertTrue(Path(cmd[cmd.index('--model')+1]).is_file())
                out.write_text('{}')
            else:
                self.fail(cmd)
        finally:
            with self.lock:
                self.active.remove(slot)

    def run_scheduler(self, stages):
        with patch.object(runner, 'STAGES', stages), \
             patch.object(runner, 'metadata', lambda p: json.loads(p.read_text())), \
             patch.object(runner, 'run_job', self.fake_job):
            runner.run_independent(self.args)

    def test_fast_lane_advances_while_slow_lane_is_still_training_and_resume_is_idempotent(self):
        self.barrier_test = threading.Event()
        self.run_scheduler((100, 500, 1000))
        self.assertTrue(self.barrier_test.is_set())
        self.assertEqual(len(self.calls), 12)
        self.assertTrue(all(cmd[1] == 'selfplay' for _, cmd in self.calls))
        for d in runner.DEPTHS:
            state = json.loads((self.args.archive/f'd{d}/status.json').read_text())
            self.assertEqual(state['phase'], 'complete')
        self.calls.clear()
        self.run_scheduler((100, 500, 1000))
        self.assertEqual(self.calls, [], 'Completed jobs must not run again')

    def test_resumes_inflight_stage_from_old_global_scheduler_without_retraining(self):
        for d in runner.DEPTHS:
            work, dst = self.args.work/f'd{d}', self.args.archive/f'd{d}'
            work.mkdir(parents=True)
            dst.mkdir()
            (work/'model.trainer.npz').write_text(json.dumps(
                dict(step=150, format=8, net=dict(hidden=256, embedding_dim=32, blocks=d))))
            for suffix in ['.npz', '.ema.npz']:
                (dst/f'model.step100{suffix}').write_text('original checkpoint')
            for variant in ['ema', 'raw']:
                (dst/f'eval_step100_{variant}.json').write_text('{}')
        (self.args.archive/'stage100.complete.json').write_text('{}')
        self.run_scheduler((100, 500))
        train = [cmd for _, cmd in self.calls if cmd[1] == 'selfplay']
        self.assertEqual(len(train), 4)
        for cmd in train:
            self.assertIn('--resume-state', cmd)
            self.assertEqual(cmd[cmd.index('--steps')+1], '500')
        for d in runner.DEPTHS:
            self.assertEqual((self.args.archive/f'd{d}/model.step100.npz').read_text(), 'original checkpoint')

    def test_lane_failure_aborts_other_lanes_instead_of_hanging(self):
        barrier = threading.Barrier(4)
        def fail_job(slot, cmd, log, workers, abort=None):
            barrier.wait(timeout=5)
            if slot == 0:
                raise RuntimeError('injected failure')
            self.assertTrue(abort.wait(5), 'Other lanes did not receive cancellation')
            raise RuntimeError('cancelled')
        with patch.object(runner, 'STAGES', (100,)), patch.object(runner, 'run_job', fail_job):
            with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                runner.run_independent(self.args)
        status = json.loads((self.args.archive/'status.json').read_text())
        self.assertEqual(status['phase'], 'failed')


if __name__ == '__main__':
    unittest.main()
