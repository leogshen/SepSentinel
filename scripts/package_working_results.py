"""Explicit publication allowlist; never include raw data, local caches or weights."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNS=['sepsis_controls_20261011','sepsis_lab_values_20261011','sepsis_long_training_20261012',
      'sepsis_long_mae_20261012','sepsis_mae_finetune_20261013','sepsis_cv_20261013','sepsis_lab_specialize_20261014']
SCRIPTS=['audit_cv_patient_precision','calculate_random_baselines','causal_control_features',
         'explain_lab_gate_recall','lab_value_ablation','run_lab_specialization','run_lab_value_experiment',
         'run_long_sepsis_training','run_long_target_mae','run_mae_finetune','run_sepsis_controls','run_sepsis_cv',
         'summarize_sepsis_cv','test_lab_specialization','test_lab_value_ablation','test_long_training',
         'test_sepsis_controls','test_sepsis_cv','build_working_report','package_working_results']

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--stage',action='store_true'); args=parser.parse_args()
    files={ROOT/'working.md',ROOT/'.gitignore'}
    files.update(ROOT/'scripts'/f'{name}.py' for name in SCRIPTS)
    files.update(ROOT/'configs'/f'{run}.json' for run in RUNS if (ROOT/'configs'/f'{run}.json').exists())
    for run in RUNS:
        folder=ROOT/'results'/run
        for parent in [folder,*sorted(folder.glob('fold_*'))]:
            if not parent.is_dir(): continue
            for path in parent.iterdir():
                if not path.is_file() or path.suffix not in ['.json','.md']: continue
                if 'calibration_curve' in path.name or path.name in ['evaluation_partial.json','gate_recall_explanation.json']: continue
                files.add(path)
    files.add(ROOT/'results/transfer_protocol/source_manifest.json')
    files.update(ROOT/'results/paper_snapshot_20261016'/name for name in ['runtime.json','report_inputs.json'])
    # Existing primary result inputs may already be tracked; explicit inclusion
    # ensures report regeneration needs no local patient data.
    files.add(ROOT/'results/sepsis_missions_adapt_20261010/evaluation.json')
    assert all(p.is_file() for p in files)
    manifest={}
    for path in sorted(files):
        relative=path.relative_to(ROOT)
        assert 'local' not in relative.parts
        assert path.suffix in ['.py','.md','.json'] or path.name=='.gitignore'
        assert path.stat().st_size<30_000_000,path
        manifest[relative.as_posix()]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    output=ROOT/'results/paper_snapshot_20261016/publication_files.json'
    output.write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(files=len(files)+1,bytes=sum(v['bytes'] for v in manifest.values()),
                          raw_data_and_weights_included=False),indent=2))
    if args.stage:
        subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}','add','--',
                        *manifest.keys(),output.relative_to(ROOT).as_posix()],cwd=ROOT,check=True)

if __name__=='__main__': main()
