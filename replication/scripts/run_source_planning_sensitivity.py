#!/usr/bin/env python3
"""One sequential, locked GPU lane for the six-run planning sensitivity study."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time


def save(path, payload):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, indent=2) + '\n')
    temp.replace(path)


def prepare(root, fold):
    results = root / 'results/planning_sensitivity_20260913'
    report = json.loads((results / 'planning_comparison.json').read_text())
    check = next(x for x in report['folds'] if x['fold'] == fold)
    assert set(check['changed_fields']) == {'patch_size'}, check
    name = f'nnUNetResEncMDevelopmentFold{fold}'
    source = results / f'{name}.json'
    destination = root / f'data/nnunet/preprocessed/Dataset501_BraTSPEDs/{name}.json'
    if destination.exists():
        assert source.read_bytes() == destination.read_bytes()
    else:
        shutil.copyfile(source, destination)
    return name, hashlib.sha256(source.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=[0, 2], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = Path.cwd()
    out = root / 'results/planning_sensitivity_20260913'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / f'fold{args.fold}_{"smoke" if args.smoke else "lane"}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plans, plan_hash = prepare(root, args.fold)
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': str(args.gpu), 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
                'nnUNet_raw': str(root / 'data/nnunet/raw'),
                'nnUNet_preprocessed': str(root / 'data/nnunet/preprocessed'),
                'nnUNet_results': str(root / 'data/nnunet/results')})
    python = str(root / '.venv/bin/python')
    if args.smoke:
        env['nnUNet_results'] = str(out / 'smoke_models')
        code = '''
import os,json,time,torch
from pathlib import Path
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgenerators.utilities.file_and_folder_operations import load_json
class SourcePlanningSmoke(nnUNetTrainer):
 def __init__(self,plans,configuration,fold,dataset_json,device=torch.device('cuda')):
  super().__init__(plans,configuration,fold,dataset_json,device)
  self.num_epochs=1
  self.num_iterations_per_epoch=1
  self.num_val_iterations_per_epoch=1
  self.save_every=1
 def do_split(self):
  training,_=super().do_split()
  return training,training[:min(16,len(training))]
base=Path(os.environ['nnUNet_preprocessed'])/'Dataset501_BraTSPEDs'
plans=load_json(str(base/'PLANS.json'))
plans['continue_training']=False
trainer=SourcePlanningSmoke(plans,'3d_fullres',FOLD,load_json(str(base/'dataset.json')),device=torch.device('cuda'))
os.environ['nnUNet_n_proc_DA']='2'
trainer.run_training()
torch.cuda.synchronize()
print('SOURCE_PLANNING_SMOKE_PASSED',json.dumps({'max_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,'max_reserved_mb':torch.cuda.max_memory_reserved()/1024**2}))
'''.replace('PLANS', plans).replace('FOLD', str(args.fold))
        # A file is needed for multiprocessing spawn; never initialize a full run from this checkpoint.
        smoke_script = out / f'smoke_fold{args.fold}.py'
        lines = code.strip().splitlines()
        # Keep imports and class top-level, guard executable statements for spawn.
        split = next(i for i, line in enumerate(lines) if line.startswith('base='))
        code = '\n'.join(lines[:split]) + '\nif __name__ == "__main__":\n' + '\n'.join(' '+line for line in lines[split:]) + '\n'
        smoke_script.write_text(code)
        subprocess.run([python, '-u', str(smoke_script)], env=env, check=True)
        return
    state_path = out / f'fold{args.fold}_worker_status.json'
    state = {'status': 'running', 'pid': os.getpid(), 'fold': args.fold, 'gpu': args.gpu,
             'plan': plans, 'plan_sha256': plan_hash, 'seeds': [2026, 2027, 2028],
             'started_unix': time.time(), 'completed_seeds': []}
    save(state_path, state)
    run_dir = root / 'results/source_planning_runs'
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = root / 'results/eval_source_planned'
    eval_dir.mkdir(parents=True, exist_ok=True)
    def execute(command, stage, logfile):
        state.update(stage=stage, child_pid=None)
        save(state_path, state)
        with logfile.open('a') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            save(state_path, state)
            code = child.wait()
        if code:
            raise RuntimeError(f'{stage} exited with code {code}; see {logfile.name}')
    try:
        for seed in state['seeds']:
            state['seed'] = seed
            status_file = run_dir / f'fold{args.fold}_seed{seed}SourceOnlyPlan.json'
            finished = status_file.exists() and json.loads(status_file.read_text()).get('status') == 'completed'
            if not finished:
                command = [python, '-u', 'scripts/run_nnunet_seeded.py', '--fold', str(args.fold), '--seed', str(seed),
                           '--plans', plans, '--trainer-suffix', 'SourceOnlyPlan', '--num-epochs', '1000',
                           '--data-processes', '12', '--device', 'cuda', '--save-probabilities', '--resume', '--status-dir', str(run_dir)]
                execute(command, 'training', root / f'logs/source_planned_fold{args.fold}_seed{seed}.log')
            prediction = root / f'data/nnunet/results/Dataset501_BraTSPEDs/nnUNetTrainerPedGlioSeed{seed}SourceOnlyPlan__{plans}__3d_fullres/fold_{args.fold}/validation'
            expected = 116 if args.fold == 0 else 80
            assert len(list(prediction.glob('*.nii.gz'))) == expected
            assert len(list(prediction.glob('*.npz'))) == expected
            result = eval_dir / f'b0source_fold{args.fold}_seed{seed}.json'
            if not result.exists():
                execute([python, 'scripts/evaluate_formal_segmentation.py', '--prediction-dir', str(prediction), '--fold', str(args.fold),
                         '--output', str(result), '--per-case-output', str(result.with_name(result.stem + '_per_case.csv'))],
                        'evaluation', root / f'logs/source_planned_eval_fold{args.fold}_seed{seed}.log')
            assert json.loads(result.read_text())['evaluated_cases'] == expected
            state['completed_seeds'].append(seed)
            save(state_path, state)
            print(f'fold={args.fold} seed={seed} training_and_evaluation_completed', flush=True)
        template = 'data/nnunet/results/Dataset501_BraTSPEDs/nnUNetTrainerPedGlioSeed{seed}SourceOnlyPlan__nnUNetResEncMDevelopmentFold{fold}__3d_fullres/fold_{fold}/validation'
        execute([python, 'scripts/ensemble_uncertainty.py', '--fold', str(args.fold), '--validation-template', template,
                 '--ensemble-dir', 'results/ensemble_source_planned', '--eval-dir', str(eval_dir),
                 '--analysis-dir', 'results/uncertainty_source_planned'], 'ensemble_and_uncertainty', root / f'logs/source_planned_ensemble_fold{args.fold}.log')
        state.update(status='completed', stage='completed', child_pid=None, finished_unix=time.time())
        save(state_path, state)
        print(f'SOURCE_PLANNING_FOLD_{args.fold}_COMPLETE', flush=True)
    except Exception as error:
        state.update(status='failed', error=str(error), finished_unix=time.time())
        save(state_path, state)
        raise


if __name__ == '__main__':
    main()
