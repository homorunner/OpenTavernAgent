#!/usr/bin/env python3
"""Bounded SSH evaluation jobs. Borrowed machines only get a unique /tmp directory.

A separate watchdog handles parent death and TTL expiration. Every result requires
an explicit cleanup check before its host can be reused.
"""
import functools
import gzip
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
import uuid

SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
       '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']

WATCHDOG = r'''
import json, os, pathlib, shutil, signal, sys, time
root=pathlib.Path(sys.argv[1]); token=sys.argv[2]; parent=int(sys.argv[3]); ttl=int(sys.argv[4])
def ident(pid):
 try:return pathlib.Path('/proc/'+str(pid)+'/stat').read_text().split()[21]
 except OSError:return None
original=ident(parent); deadline=time.monotonic()+ttl
while root.exists():
 if ident(parent)!=original or time.monotonic()>deadline:
  if ident(parent)==original:
   os.kill(parent, signal.SIGTERM); time.sleep(10)
  if root.exists() and (root/'owner').read_text()==token:
   try:
    active=json.loads((root/'active.json').read_text()); pid=active['pid']
    if ident(pid)==active['start']:os.killpg(pid, signal.SIGKILL)
   except (OSError,ValueError,KeyError):pass
   shutil.rmtree(root)
  break
 time.sleep(3)
'''

BOOTSTRAP = r'''
import gzip, json, os, pathlib, shutil, signal, subprocess, sys, tarfile, time
root=pathlib.Path(sys.argv[1]); token=sys.argv[2]; ttl=int(sys.argv[3]); watchdog_code=sys.argv[4]
assert root.parent==pathlib.Path('/tmp') and root.name.startswith('tavernagent-depth-eval-') and token in root.name
child=None; guard=None; owned=False
def stop(signum,frame):raise SystemExit(128+signum)
for sig in (signal.SIGHUP,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
try:
 os.umask(0o077); root.mkdir(); owned=True; (root/'owner').write_text(token)
 guard=subprocess.Popen(['python3','-B','-c',watchdog_code,str(root),token,str(os.getpid()),str(ttl)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
 with gzip.GzipFile(fileobj=sys.stdin.buffer,mode='rb') as compressed, tarfile.open(fileobj=compressed,mode='r|') as archive:
  for member in archive:
   assert member.isfile() and '/' not in member.name and member.name not in ('.','..')
   with archive.extractfile(member) as source, (root/member.name).open('wb') as dest:shutil.copyfileobj(source,dest)
 (root/'tavernagent').chmod(0o700)
 child=subprocess.Popen(['python3','-B',str(root/'worker.py')],cwd=root,stdin=subprocess.DEVNULL,start_new_session=True)
 (root/'active.json').write_text(json.dumps({'pid':child.pid,'start':pathlib.Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21]}))
 try:code=child.wait(timeout=ttl-30)
 except subprocess.TimeoutExpired:code=124
 sys.exit(code)
finally:
 for sig in (signal.SIGHUP,signal.SIGTERM,signal.SIGINT):signal.signal(sig,signal.SIG_IGN)
 if child is not None:
  try:os.killpg(child.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   child.wait()
 if owned and root.exists() and (root/'owner').read_text()==token:shutil.rmtree(root)
 if guard is not None:
  guard.terminate(); guard.wait(timeout=10)
 print(json.dumps({'event':'cleanup','root':str(root),'absent':not root.exists()}),file=sys.stderr,flush=True)
'''


WORKER = r'''
import json,os,pathlib,subprocess,sys,time
job=json.loads(pathlib.Path('job.json').read_text())
env=dict(os.environ,RAYON_NUM_THREADS=str(job['workers']))
start=time.monotonic()
with open('command.log','w') as log:
 p=subprocess.run(['./tavernagent',*job['args']],env=env,stdout=log,stderr=subprocess.STDOUT)
result=json.loads(pathlib.Path('result.json').read_text()) if p.returncode==0 else None
print(json.dumps(dict(result=result,seconds=time.monotonic()-start,exit_code=p.returncode,
 log=pathlib.Path('command.log').read_text()[-12000:])),flush=True)
sys.exit(p.returncode)
'''


def tar_member(stream, name, data):
    info = tarfile.TarInfo(name)
    info.size, info.mode = len(data), 0o600
    stream.write(info.tobuf())
    stream.write(data)
    stream.write(b'\0' * ((-len(data)) % 512))


@functools.lru_cache(maxsize=1)
def binary_prefix(binary):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=1) as stream:
        tar_member(stream, 'tavernagent', Path(binary).read_bytes())
    return buf.getvalue()


def cleanup(host, remote, token):
    """Recover only a recorded job directory, checking ownership and PID identity."""
    code = r'''
from pathlib import Path
import os,json,signal,shutil,sys
root=Path(sys.argv[1]); token=sys.argv[2]
assert root.parent==Path('/tmp') and root.name.startswith('tavernagent-depth-eval-') and token in root.name
if root.exists():
 assert (root/'owner').read_text()==token
 try:
  active=json.loads((root/'active.json').read_text()); pid=active['pid']
  if Path(f'/proc/{pid}/stat').read_text().split()[21]==active['start']:
   os.killpg(pid,signal.SIGKILL)
 except (FileNotFoundError,ProcessLookupError):pass
 shutil.rmtree(root,ignore_errors=True)
assert not root.exists()
'''
    subprocess.run([*SSH, host, shlex.join(['python3', '-B', '-c', code, remote, token])],
                   check=True, capture_output=True, timeout=30)


def run_remote(job, host, binary, directory, ttl=2400):
    directory.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    remote = '/tmp/tavernagent-depth-eval-' + token
    assignment = dict(host=host['host'], remote=remote, token=token, started=time.time(), cleanup_verified=False)
    assignment_path = directory/'assignment.json'
    assignment_path.write_text(json.dumps(assignment, indent=2))
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=1) as stream:
        tar_member(stream, 'worker.py', WORKER.encode())
        tar_member(stream, 'job.json', json.dumps(dict(args=job['args'], workers=host['cpus'])).encode())
        for name, path in job['inputs'].items():
            if '/' in name or name in ('.', '..'):
                raise ValueError('Inputs must have flat names')
            tar_member(stream, name, Path(path).read_bytes())
        stream.write(b'\0' * 1024)
    cmd = [*SSH, host['host'], shlex.join(['python3', '-u', '-B', '-c', BOOTSTRAP,
                                        remote, token, str(ttl), WATCHDOG])]
    failure = None
    result = None
    try:
        with (directory/'response.json').open('wb') as output, (directory/'remote.log').open('w') as log:
            with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=output, stderr=log) as child:
                try:
                    child.stdin.write(binary_prefix(str(binary)))
                    child.stdin.write(buf.getvalue())
                    child.stdin.close()
                    code = child.wait(timeout=ttl+120)
                except BaseException:
                    child.kill()
                    child.wait()
                    raise
        response = json.loads((directory/'response.json').read_text())
        if code or response['exit_code'] or response['result'] is None:
            raise RuntimeError(f'Remote job failed: {response.get("log", "")[-1500:]}')
        result = response
    except BaseException as exc:
        failure = exc
    finally:
        try:
            # Check ordinary finally cleanup first; only recover if needed.
            check = subprocess.run([*SSH, host['host'], shlex.join(['test', '!', '-e', remote])],
                                   capture_output=True, timeout=30)
            if check.returncode:
                cleanup(host['host'], remote, token)
            assignment.update(cleanup_verified=True, finished=time.time())
        except Exception as exc:
            assignment.update(cleanup_error=str(exc), finished=time.time())
            failure = RuntimeError('Borrowed host cleanup could not be verified; do not reuse it')
        assignment_path.write_text(json.dumps(assignment, indent=2))
    if failure:
        raise failure
    return result
