"""One isolated 64-env/20-update trial. No API keys, network installs or retries."""
import argparse,hashlib,importlib.metadata,json,os,re,signal,subprocess,sys,time,urllib.parse,urllib.request,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/toe_smoke'
MOTION=ROOT/'motions/gui00528900_toe_vertical_parallel_candidate/motion.npz'
MOTION_SHA='68e6d580354a86caeed1b94e7869475fb3220c1fa4dc95894057301aed62fb35'
MAX_SECONDS=1200

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def event(name,**kw): print(json.dumps({'event':name,**kw},sort_keys=True),flush=True)

def preflight(expected_commit=None):
    manifest=json.loads((ROOT/'flux/input_manifest.json').read_text())
    for v in manifest:
        p=ROOT/v['path']
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=v['bytes'] or digest(p)!=v['sha256']:raise ValueError('Input manifest mismatch: '+v['path'])
        if p.read_bytes()[:45].startswith(b'version https://git-lfs.github.com'):raise ValueError('LFS pointer')
        if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
    if digest(MOTION)!=MOTION_SHA:raise ValueError('Motion digest mismatch')
    if expected_commit:
        got=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        if got!=expected_commit:raise ValueError('Unexpected Git commit')
    return {'input_files':len(manifest),'input_bytes':sum(v['bytes'] for v in manifest),'motion_sha256':MOTION_SHA}

def run_bounded(command,seconds,log_path,env=None,heartbeat=30):
    with open(log_path,'w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        end=time.monotonic()+seconds;next_beat=time.monotonic()+heartbeat
        while child.poll() is None:
            if time.monotonic()>=end:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=2)
                return 124
            if time.monotonic()>=next_beat:
                event('worker_heartbeat',remaining_seconds=round(end-time.monotonic()))
                next_beat=time.monotonic()+heartbeat
            time.sleep(min(0.2,max(0.01,end-time.monotonic())))
        return child.returncode

def worker():
    if sys.version_info[:2]!=(3,11):raise RuntimeError('Requires target CPython 3.11')
    event('source_preflight_verified',**preflight(os.environ['TOE_COMMIT']))
    deps=ROOT/'.flux_deps';deps.mkdir(exist_ok=False)
    # The image remains the provider of torch/CUDA/numpy/IsaacSim/IsaacLab.
    # Add only missing wheel distributions, never blindly upgrade the image.
    missing=[]
    for p in (ROOT/'flux/wheelhouse').glob('*.whl'):
        dist=p.name.split('-')[0]
        try:importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:missing.append(str(p))
    if missing:
        subprocess.run([sys.executable,'-m','pip','--isolated','--disable-pip-version-check','install','--no-index','--no-cache-dir','--no-deps','--target',str(deps),*missing],check=True,timeout=90)
    subprocess.run([sys.executable,'-m','pip','--isolated','--disable-pip-version-check','install','--no-index','--no-cache-dir','--no-deps','--no-build-isolation','--target',str(deps),str(ROOT/'flux/vendor')],check=True,timeout=60)
    sys.path.insert(0,str(deps));sys.path.insert(0,str(ROOT/'source/limx_rl_lab'));sys.path.insert(0,str(ROOT))
    os.environ['PYTHONPATH']=os.pathsep.join([str(deps),str(ROOT/'source/limx_rl_lab'),str(ROOT),os.environ.get('PYTHONPATH','')])
    from packaging.requirements import Requirement
    for dist in ['rsl-rl-lib','tensordict','tensorboard','onnx','GitPython','prettytable']:
        for text in importlib.metadata.requires(dist) or []:
            req=Requirement(text)
            if req.marker and not req.marker.evaluate({'extra':''}):continue
            v=importlib.metadata.version(req.name)
            if req.specifier and v not in req.specifier:raise RuntimeError('Dependency constraint failed: '+req.name)
    import numpy as np,torch
    if torch.__version__.split('+')[0]!='2.7.0' or int(np.__version__.split('.')[0])>=2:raise RuntimeError('Unexpected image Torch/NumPy')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Requires exactly one visible GPU')
    assert torch.ones(4,device='cuda').sum().item()==4
    event('dependencies_verified',python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,rsl_rl=importlib.metadata.version('rsl-rl-lib'),cuda=torch.version.cuda)
    with np.load(MOTION,allow_pickle=False) as data:
        assert data['joint_pos'].shape==(187,29)
        for k in data.files:
            if not np.isfinite(data[k]).all():raise ValueError('Nonfinite motion')
    # Physics/runtime validation happens only after Isaac starts; this is not a quality claim.
    import runpy
    sys.path.insert(0,str(ROOT/'scripts/rsl_rl'))
    sys.argv=[str(ROOT/'scripts/rsl_rl/train.py'),'--task','LimX-HU-D04-01-Flat-BeyondMimic','--motion_file',str(MOTION),'--num_envs','64','--max_iterations','20','--seed','42','--run_name','toe_vertical_parallel_smoke','--logger','tensorboard','--headless']
    event('simulator_initializing')
    runpy.run_path(str(ROOT/'scripts/rsl_rl/train.py'),run_name='__main__')

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def seal_and_upload(url,headers,child_code,started):
    parsed=urllib.parse.urlsplit(url)
    if parsed.scheme!='https' or not (parsed.hostname=='storage.googleapis.com' or str(parsed.hostname).endswith('.storage.googleapis.com')):raise ValueError('Not a scoped GCS output')
    if any(k.lower()!='content-type' and not k.lower().startswith('x-goog-') for k in headers):raise ValueError('Unsafe upload header')
    files=[p for p in OUT.glob('*.json*') if p.is_file()]
    checkpoints=sorted((ROOT/'logs').rglob('model*.pt'),key=lambda p:p.stat().st_mtime) if (ROOT/'logs').exists() else []
    if checkpoints:
        final=next((p for p in checkpoints if p.name=='model_final_20_updates.pt'),checkpoints[-1])
        files.append(final)
        files.extend(p for p in (final.parent/'params').glob('*') if p.is_file() and p.suffix in ['.yaml','.py'])
        files.extend(p for p in final.parent.glob('events.out.tfevents.*') if p.is_file())
    status={'worker_exit_code':child_code,'elapsed_seconds':round(time.monotonic()-started,2),'input_sha256':digest(MOTION),'input_unchanged':digest(MOTION)==MOTION_SHA,'checkpoint_present':bool(checkpoints),'commit':os.environ['TOE_COMMIT'],'automatic_retries':0}
    (OUT/'worker_status.json').write_text(json.dumps(status,indent=2));files.append(OUT/'worker_status.json')
    if sum(p.stat().st_size for p in files)>100*1024*1024:raise ValueError('Output exceeds 100 MiB')
    bundle=OUT/'toe_smoke_bundle.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            if p.is_symlink() or not p.resolve().is_relative_to(ROOT):raise ValueError('Unsafe output')
            z.write(p,str(p.relative_to(ROOT)))
    request=urllib.request.Request(url,data=bundle.read_bytes(),headers=headers,method='PUT')
    with urllib.request.build_opener(NoRedirect()).open(request,timeout=25) as response:
        if response.status not in (200,201):raise RuntimeError('Output PUT rejected')
    event('output_uploaded',sha256=digest(bundle),bytes=bundle.stat().st_size,**status)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',action='store_true');parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--commit');parser.add_argument('--upload-url');parser.add_argument('--upload-headers',default='{}')
    args=parser.parse_args()
    if args.self_test:
        import tempfile
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            assert run_bounded([sys.executable,'-c','print(123)'],3,Path(d)/'ok')==0
            assert run_bounded([sys.executable,'-c','import time;time.sleep(10)'],0.3,Path(d)/'stop')==124
        event('watchdog_tests_passed');return
    if args.worker:worker();return
    event('cpu_preflight_verified',**preflight(args.commit))
    if not args.execute:return
    if not args.commit or not args.upload_url:raise ValueError('Pinned commit and scoped output PUT required')
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=False)
    launchers=[Path('/isaac-sim/python.sh'),Path('/workspace/isaaclab/_isaac_sim/python.sh')]
    launcher=next((p for p in launchers if p.is_file() and os.access(p,os.X_OK)),None)
    if launcher is None:raise RuntimeError('No supported Isaac Python launcher found')
    os.environ.update({'TOE_SMOKE_BOUNDED':'1','TOE_COMMIT':args.commit,'TOE_MOTION_SHA256':MOTION_SHA,'TOE_OUTPUT':str(OUT),'TOE_SOFT_DEADLINE':str(time.time()+1090),'PYTHONUNBUFFERED':'1','ACCEPT_EULA':'Y','PRIVACY_CONSENT':'Y','CUDA_VISIBLE_DEVICES':'0','WANDB_MODE':'disabled'})
    event('authorized_trial_begin',num_envs=64,max_iterations=20,worker_limit_seconds=1200,automatic_retries=0)
    code=run_bounded([str(launcher),str(Path(__file__).resolve()),'--worker'],1130,OUT/'worker-private.log',env=os.environ.copy())
    # Local private simulator logs are not uploaded. Surface only a short scrubbed failure tail.
    if code:
        tail=(OUT/'worker-private.log').read_text(errors='replace')[-6000:]
        tail=re.sub(r'https?://\S+','[URL omitted]',tail)
        event('worker_failed',exit_code=code,sanitized_tail=tail)
    seal_and_upload(args.upload_url,json.loads(args.upload_headers),code,started)
    if code or not (OUT/'completion.json').is_file():raise SystemExit(code or 1)
    event('entrypoint_complete',elapsed_seconds=round(time.monotonic()-started,2))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        event('entrypoint_failed',error_type=type(exc).__name__)
        raise SystemExit(1)