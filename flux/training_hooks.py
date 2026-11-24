"""Auditable counters around the unmodified PPO update; no reward/physics changes."""
import hashlib,json,math,os,time
from pathlib import Path

def event(name,**kw):
    print(json.dumps({'event':name,**kw},sort_keys=True),flush=True)

def install(runner):
    assert runner.env.num_envs==64
    assert runner.num_steps_per_env==24
    runner.save_interval=5
    runner._toe_updates=0
    update=runner.alg.update
    def checked_update(*args,**kwargs):
        if runner._toe_updates>=20 or time.time()>float(os.environ['TOE_SOFT_DEADLINE']):
            raise TimeoutError('Authorized PPO/time boundary reached')
        losses=update(*args,**kwargs)
        runner._toe_updates+=1
        vals={k:float(v) for k,v in losses.items()}
        if not all(math.isfinite(v) for v in vals.values()): raise ValueError('Nonfinite PPO losses')
        row={'event':'training_update_completed','completed_updates':runner._toe_updates,'losses':vals}
        with open(Path(os.environ['TOE_OUTPUT'])/'updates.jsonl','a') as f:f.write(json.dumps(row)+'\n')
        event('training_update_completed',completed_updates=runner._toe_updates,losses=vals)
        return losses
    runner.alg.update=checked_update
    save=runner.save
    def checked_save(path,*args,**kwargs):
        result=save(path,*args,**kwargs)
        p=Path(path)
        event('checkpoint_saved',file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),completed_updates=runner._toe_updates)
        return result
    runner.save=checked_save
    event('ppo_hooks_installed',num_envs=64,iterations_limit=20)

def complete(runner):
    if runner._toe_updates!=20:raise RuntimeError('Did not complete exactly twenty PPO updates')
    final=Path(runner.log_dir)/'model_final_20_updates.pt'
    runner.save(str(final))
    record={'completed_updates':20,'num_envs':64,'num_steps_per_env':24,'total_training_transitions':30720,'checkpoint':str(final.relative_to(Path.cwd())),'checkpoint_bytes':final.stat().st_size,'checkpoint_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'input_sha256':os.environ['TOE_MOTION_SHA256'],'commit':os.environ['TOE_COMMIT'],'stop_reason':'authorized_20_updates_complete','dynamic_policy_quality_validated':False}
    (Path(os.environ['TOE_OUTPUT'])/'completion.json').write_text(json.dumps(record,indent=2))
    event('training_complete',**record)