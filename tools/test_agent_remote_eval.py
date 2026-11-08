import gzip
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
import uuid

import agent_remote_job as remote
from agent_remote_eval import jobs_for, league_jobs, STAGES, DEPTHS


class PlanningTests(unittest.TestCase):
    def test_jobs_wait_only_for_their_own_inputs_and_cover_every_100_rounds(self):
        index=dict(checkpoints={'d6/model.step100.ema.npz':'path'},matched=None)
        self.assertEqual(len(jobs_for(index)),2)
        index['checkpoints']['d8/model.step100.ema.npz']='path2'
        jobs=jobs_for(index)
        self.assertEqual([j['kind'] for j in jobs].count('arena'),0)
        self.assertEqual([j['kind'] for j in jobs].count('eval'),4)
        index['checkpoints']={f'd{d}/model.step{s}{suffix}':'path'
            for d in DEPTHS for s in STAGES for suffix in ['.ema.npz','.npz']}
        index['matched']={'steps':{str(d):3000 for d in DEPTHS}}
        jobs=jobs_for(index)
        self.assertEqual(len(jobs),480)
        self.assertEqual(len({j['id'] for j in jobs}),480)
        self.assertEqual({j['step'] for j in jobs if j['kind']=='eval'},set(range(100,3001,100)))

    def test_teacher_backfill_and_idle_batches_are_independent(self):
        from agent_eval_precision import extras, merge
        from agent_remote_eval import complete
        index = {'checkpoints': {f'd6/model.step{s}.ema.npz': 'path' for s in (100, 200)}}
        jobs = jobs_for(index)
        self.assertEqual(len({j['relative'] for j in jobs}), 4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for j in jobs:
                teacher = j['opponent'] == 'teacher'
                games = int(j['args'][j['args'].index('--games')+1])
                self.assertEqual(games, 2048 if teacher or j['step'] != 100 else 512)
                self.assertEqual('--opponent' in j['args'], teacher)
                target = root/'results'/j['relative']
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps({'opponent': j['opponent'], 'summary': {'network': {'n': games}}}))
            batches = extras(jobs, root, set(), complete)
            self.assertEqual(len(batches), 2)
            self.assertEqual({j['opponent'] for j in batches}, {'scripted', 'teacher'})
            self.assertTrue(all(j['step'] == 200 for j in batches))
            for j in batches:
                self.assertEqual(j['args'][j['args'].index('--games')+1], '512')
                self.assertGreater(int(j['args'][j['args'].index('--seed')+1]), 61002048)
            remaining = extras(jobs, root, {batches[0]['id']}, complete)
            self.assertEqual(len(remaining), 1)
            self.assertNotEqual(remaining[0]['opponent'], batches[0]['opponent'])
        with self.assertRaisesRegex(AssertionError, 'Different evaluation opponents'):
            merge({'placements': {}, 'opponent': 'teacher'}, {'opponent': 'scripted'}, 'bad')

    def test_league_uses_every_checkpoint_and_disjoint_batch_seeds(self):
        index={'checkpoints':{f'd{d}/model.step100{suffix}':'path'
            for d in DEPTHS for suffix in ['.ema.npz','.npz']}}
        with tempfile.TemporaryDirectory() as tmp:
            jobs=league_jobs(index,Path(tmp),{},3)
        self.assertEqual(len(jobs),3)
        seeds=[]
        for job in jobs:
            self.assertEqual(job['kind'],'league')
            self.assertEqual(len(job['inputs']),8)
            args=job['args']
            models=[args[i+1] for i,x in enumerate(args) if x=='--model']
            self.assertEqual(len(set(models)),8)
            self.assertEqual(args[args.index('--tables')+1],'tables.json')
            self.assertEqual(len(job['tables']),128)
            seeds.append(args[args.index('--seed')+1])
        self.assertEqual(len(set(seeds)),3)


class BorrowedWorkspaceTests(unittest.TestCase):
    def launch(self, command):
        token=uuid.uuid4().hex
        root=Path('/tmp')/('tavernagent-depth-eval-'+token)
        buf=io.BytesIO()
        with gzip.GzipFile(fileobj=buf,mode='wb') as stream:
            remote.tar_member(stream,'worker.py',remote.WORKER.encode())
            remote.tar_member(stream,'job.json',json.dumps(dict(args=['-c',command],workers=1)).encode())
            stream.write(b'\0'*1024)
        proc=subprocess.Popen(['python3','-u','-B','-c',remote.BOOTSTRAP,str(root),token,'90',remote.WATCHDOG],
                              stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        proc.stdin.write(remote.binary_prefix('/bin/sh')+buf.getvalue())
        proc.stdin.close()
        proc.stdin=None
        return root,proc

    def test_success_and_failure_both_remove_all_remote_files(self):
        for command,code in [("echo '{\"score\":0.5}' > result.json",0),('exit 7',7)]:
            root,proc=self.launch(command)
            stdout,stderr=proc.communicate(timeout=15)
            self.assertEqual(proc.returncode,code,stderr.decode())
            self.assertFalse(root.exists())
            self.assertTrue(json.loads(stderr.decode().splitlines()[-1])['absent'])
            self.assertEqual(json.loads(stdout)['exit_code'],code)

    def test_watchdog_cleans_up_after_controller_is_killed(self):
        root,proc=self.launch('sleep 60')
        deadline=time.monotonic()+5
        while not (root/'active.json').exists() and time.monotonic()<deadline:
            time.sleep(.05)
        self.assertTrue((root/'active.json').exists())
        proc.kill()
        proc.wait(timeout=5)
        deadline=time.monotonic()+8
        while root.exists() and time.monotonic()<deadline:
            time.sleep(.1)
        self.assertFalse(root.exists(),'Watchdog did not remove abandoned workspace')
        proc.stdout.close()
        proc.stderr.close()


if __name__=='__main__':
    unittest.main()
