from datetime import datetime, timezone
from pathlib import Path
import json, os, subprocess, sys, time
root=Path(__file__).resolve().parents[2]
dest=Path(__file__).resolve().parent
env=dict(os.environ, HF_HOME=str(root/'runs/huggingface'), HF_HUB_CACHE=str(root/'runs/huggingface/hub'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_ENABLE_HF_TRANSFER='0', CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
state={'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running','jobs':[]}
def save():
 (dest/'timing-queue.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 for job,size in [('clock-window','smoke'),('clock-window','baseline'),('clock-stability','baseline')]:
  cmd=[sys.executable,'scripts/run_research_job.py','--node','h100-80-02','--gpu','0','--job',job,'--size',size,'--model-lock',str(root/'runs/campaign-lock/model-refs.json')]
  entry={'job':job,'size':size,'command':cmd,'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running'}
  state['jobs'].append(entry); save()
  print('Starting',job,size,flush=True)
  started=time.monotonic()
  with (dest/f'{job}-{size}-launcher.log').open('x') as log:
   result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
  entry.update(returncode=result.returncode,status='complete' if result.returncode==0 else 'failed',elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
  save()
  print('Finished',job,size,'returncode',result.returncode,flush=True)
  if result.returncode: raise RuntimeError(f'{job} failed')
 state['status']='complete'
except BaseException as exc:
 state.update(status='failed',error=str(exc)); raise
finally:
 state['finished_utc']=datetime.now(timezone.utc).isoformat(); save()
