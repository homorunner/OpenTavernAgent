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
from agent_remote_eval import jobs_for, STAGES, DEPTHS


class PlanningTests(unittest.TestCase):
    def test_jobs_wait_only_for_their_own_inputs_and_cover_every_100_rounds(self):
        index=dict(checkpoints={'d6/model.step100.ema.npz':'path'},matched=None)
        self.assertEqual(len(jobs_for(index)),1)
        index['checkpoints']['d8/model.step100.ema.npz']='path2'
        jobs=jobs_for(index)
        self.assertEqual([j['kind'] for j in jobs].count('arena'),1)
        self.assertEqual([j['kind'] for j in jobs].count('eval'),2)
        index['checkpoints']={f'd{d}/model.step{s}{suffix}':'path'
            for d in DEPTHS for s in STAGES for suffix in ['.ema.npz','.npz']}
        index['matched']={'steps':{str(d):3000 for d in DEPTHS}}
        jobs=jobs_for(index)
        self.assertEqual(len(jobs),426)
        self.assertEqual(len({j['id'] for j in jobs}),426)
        self.assertEqual({j['step'] for j in jobs if j['kind']=='eval'},set(range(100,3001,100)))


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
