from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, os, subprocess, sys, time
root=Path(__file__).resolve().parents[2]
dest=Path(__file__).resolve().parent
env=dict(os.environ, HF_HOME=str(root/'runs/huggingface'), HF_HUB_CACHE=str(root/'runs/huggingface/hub'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_ENABLE_HF_TRANSFER='0', CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
state={'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running','jobs':[]}
def save():
 (dest/'frozen-queue.json').write_text(json.dumps(state,indent=2)+'\n')
def run(name,args):
 cmd=[sys.executable,'-m','experiments.exp_frozen_replay_gpu',*args]
 entry={'name':name,'command':cmd,'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running'}
 state['jobs'].append(entry); save()
 started=time.monotonic()
 print('Starting',name,flush=True)
 with (dest/f'{name}.log').open('x') as log:
  result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 entry.update(returncode=result.returncode,status='complete' if result.returncode==0 else 'failed',elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
 save()
 print('Finished',name,'returncode',result.returncode,flush=True)
 if result.returncode: raise RuntimeError(f'{name} failed')
def capture(name,n,t):
 run(name,['capture','--node','h100-80-02','--prompts',str(n),'--tokens',str(t),'--output',str(dest/name)])
 return dest/name/'capture.json.gz'
def replay(name,path):
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 run(name,['replay','--node','h100-80-02','--capture',str(path),'--capture-sha256',digest,'--output',str(dest/name)])
save()
try:
 smoke=capture('capture-smoke',4,16)
 replay('replay-smoke',smoke)
 incoming=root/'runs/frozen-20261018/capture-baseline/capture.json.gz'
 if incoming.exists():
  expected='b63b753407f5395d4910730db661f9c6c97d95fcfb9aca7e9c569a63a2ee2d6a'
  if hashlib.sha256(incoming.read_bytes()).hexdigest()!=expected: raise RuntimeError('Incoming A100 capture hash mismatch')
  state['capture_origin']='archived A100 capture'; save()
  replay('replay-a100-capture',incoming)
 else:
  state['capture_origin']='new independent H100 capture; unpaired with A100'; save()
  baseline=capture('capture-baseline',80,256)
  replay('replay-baseline',baseline)
 cmd=[sys.executable,'scripts/run_research_job.py','--node','h100-80-02','--gpu','0','--job','clock-stability','--size','baseline','--model-lock',str(root/'runs/campaign-lock/model-refs.json')]
 entry={'name':'clock-stability-delayed','command':cmd,'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running'}
 state['jobs'].append(entry); save()
 print('Starting delayed timing repeat',flush=True)
 started=time.monotonic()
 with (dest/'clock-stability-delayed-launcher.log').open('x') as log:
  result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 entry.update(returncode=result.returncode,status='complete' if result.returncode==0 else 'failed',elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat()); save()
 if result.returncode: raise RuntimeError('Delayed timing repeat failed')
 state['status']='complete'
except BaseException as exc:
 state.update(status='failed',error=str(exc)); raise
finally:
 state['finished_utc']=datetime.now(timezone.utc).isoformat(); save()
