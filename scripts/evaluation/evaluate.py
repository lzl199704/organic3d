"""One command for the paper's evaluation of one model: the harness runs and their metrics, in order.

    # encoder (Table 1): every test, or a subset
    python scripts/evaluation/evaluate.py encoder --model organic3d_enc --ckpt zllhhhh/organic3d-encoder --name enc4_organic3d \
        [--tests mlc,ll,side,lesion,tumour,le]
    # VLM (Tables 2 and 3)
    python scripts/evaluation/evaluate.py vlm --model organic3d_vlm --ckpt zllhhhh/organic3d-vlm --name vlm4_organic3d \
        [--tests answers,reports]
    # a released model in its own environment: harness steps use --model_python, metric steps this python
    python scripts/evaluation/evaluate.py encoder --model hlip --name HLIP --model_python <hlip env>/bin/python
    # free-text report metrics need the report environment (requirements-report.txt, Java 17 on PATH)
    python scripts/evaluation/evaluate.py vlm ... --report_python <report env>/bin/python
    python scripts/evaluation/evaluate.py encoder ... --dry_run      # print the commands only
    python scripts/evaluation/evaluate.py                            # no arguments: asks for them

Tests (encoder):  mlc     MLC-AUROC, MP and OE on CT-RATE, RAD-ChestCT and Merlin (fat and soft-tissue fills)
                  ll      lung lobe erasure hit rate (CT-RATE)
                  side    laterality accuracy (CT-RATE, Merlin)
                  lesion  lesion AUROC on the 464 radiologist-annotated ReXGroundingCT scans
                  tumour  zero-shot tumour detection on AbdomenAtlas 3.0
                  le      lesion-erasure hit rate (ReXGroundingCT, AbdomenAtlas 3.0)
Tests (VLM):      answers answer AUROC / BACC / F1, MP and OEH on the full CT-RATE and Merlin test sets, ReXGroundingCT
                  reports free-text reports (F1, CRG, GREEN, RadGraph F1) and finding-based reports (F1, IA-F1)
Needs the prepared data and harness sets (docs/data_preparation.md). Outputs: {work_root}/extaudit/out*/<name>/ (scores)
and {work_root}/eval/ (metrics).
"""
import argparse
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, '..', '..')
ENC_TESTS = ['mlc', 'll', 'side', 'lesion', 'tumour', 'le']
VLM_TESTS = ['answers', 'reports']
REPORT_HEAD = {'REPORT_HEAD_CTRATE': 'Trachea', 'REPORT_HEAD_MERLIN': 'Lower thorax:'}


def steps(a, root):
    """-> list of (env vars, python, args): harness steps run with the model's python, metric steps with this one."""
    mp, me, mr = a.model_python, sys.executable, a.report_python
    h = ['--model', a.model, '--name', a.name, '--root', root] + (['--ckpt', a.ckpt] if a.ckpt else [])
    env = ['--env', a.env]
    out = []
    if a.kind == 'encoder':
        core = ['-m', 'extaudit.core'] + h
        for t in a.tests:
            if t == 'mlc':
                for d in ('ctrate', 'radchest', 'merlin'):
                    out += [({}, mp, core + ['--dataset', d, '--fills', 'fat,soft']),
                            ({}, me, ['scripts/metrics/extaudit_metrics.py'] + env + ['--name', a.name, '--dataset', d])]
            elif t == 'll':
                out += [({}, mp, core + ['--task', 'chest']), ({}, me, ['scripts/metrics/chest_harness_metrics.py', '--names', a.name])]
            elif t == 'side':
                out += [({'SIDE_SET': d}, mp, core + ['--task', 'side']) for d in ('ctrate', 'merlin')]
                out += [({}, me, ['scripts/metrics/side_metrics.py', '--names', a.name])]
            elif t == 'lesion':
                out += [({}, mp, core + ['--task', 'grounding', '--rex_set', 'ctrate_rex_set_v2.parquet',
                                         '--out_tag', 'grounding_v2']),
                        ({}, me, ['scripts/metrics/lesion_cls_metrics.py', '--encoders', a.name, '--vlms', ''])]
            elif t == 'tumour':
                out += [({}, mp, core + ['--task', 'aa3_det']), ({}, me, ['scripts/metrics/lesion_det_metrics.py', '--names', a.name])]
            elif t == 'le':
                for d in ('rex', 'aa3'):
                    out += [({'LE_SET': d}, mp, core + ['--task', 'lesion_erasure']),
                            ({}, me, ['scripts/metrics/heal_metrics.py', '--ds', d, '--names', a.name])]
    else:
        for t in a.tests:
            if t == 'answers':
                vc = ['-m', 'extaudit.vlm_core'] + h
                out += [({}, mp, vc + ['--dataset', d])
                        for d in ('ctrate', 'merlin')]
                out += [({}, mp, vc + ['--dataset', 'ctrate', '--set_file', 'rex_heldout_set.parquet', '--clean_only',
                                       '--out_tag', 'rex_heldout']),
                        ({}, me, ['scripts/metrics/vlm_metrics.py'] + env + ['--name', a.name]),
                        ({}, me, ['scripts/metrics/finding_reports.py', '--names', a.name]),
                        ({}, me, ['scripts/metrics/rex_vlm_metrics.py', '--mode', 'clean', '--names', a.name])]
            elif t == 'reports':
                out += [(REPORT_HEAD, mp, ['-m', 'extaudit.vlm_report'] + h + ['--dataset', d])
                        for d in ('ctrate', 'merlin')]
                out += [({}, mr, ['scripts/metrics/report_metrics_std.py', '--names', a.name]),
                        ({}, mr, ['scripts/metrics/report_green.py', '--names', a.name, '--max_length', '4096']),
                        ({}, mr, ['scripts/metrics/report_green.py', '--aggregate'])]
    return out


def ask(a):
    """Interactive mode: fill in the arguments that were not given."""
    def q(msg, default=None):
        r = input(f'{msg}{f" [{default}]" if default else ""}: ').strip()
        return r or default
    a.kind = q('Model kind (encoder / vlm)', 'encoder')
    a.model = q('Adapter module in extaudit/models or extaudit/vlm_models', 'organic3d_enc' if a.kind == 'encoder'
                else 'organic3d_vlm')
    a.ckpt = q('Checkpoint (adapter-specific; empty for none)', None)
    a.name = q('Output name', a.model)
    tests = ENC_TESTS if a.kind == 'encoder' else VLM_TESTS
    a.tests = q(f'Tests ({",".join(tests)})', 'all')
    a.model_python = q("The model environment's python", sys.executable)
    if a.kind == 'vlm':
        a.report_python = q("The report environment's python (requirements-report.txt)", sys.executable)
    a.dry_run = q('Only print the commands? (y / n)', 'n').lower().startswith('y')
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n\n', 1)[1])
    ap.add_argument('kind', nargs='?', choices=['encoder', 'vlm'])
    ap.add_argument('--model', help='adapter module (extaudit/models/<model>.py or extaudit/vlm_models/<model>.py)')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--name', help='output name')
    ap.add_argument('--tests', default='all')
    ap.add_argument('--env', default='configs/env/local.yaml')
    ap.add_argument('--model_python', default=sys.executable, help="python of the model's environment")
    ap.add_argument('--report_python', default=sys.executable,
                    help='python of the report environment (requirements-report.txt; Java 17 on PATH for METEOR)')
    ap.add_argument('--dry_run', action='store_true')
    a = ap.parse_args()
    if a.kind is None:
        a = ask(a)
    if not (a.model and a.name):
        ap.error('--model and --name are required')
    allt = ENC_TESTS if a.kind == 'encoder' else VLM_TESTS
    a.tests = allt if a.tests == 'all' else a.tests.split(',')
    bad = set(a.tests) - set(allt)
    if bad:
        ap.error(f'unknown tests {sorted(bad)}; choose from {allt}')
    os.environ.setdefault('ORGANIC3D_ENV', os.path.abspath(os.path.join(REPO, a.env)))
    sys.path.insert(0, REPO)
    from organic.workroot import work_root
    root = os.path.join(work_root(), 'extaudit')
    for i, (env, py, args) in enumerate(steps(a, root), 1):
        line = ' '.join([f'{k}={shlex.quote(v)}' for k, v in env.items()] + [shlex.quote(py)] + [shlex.quote(x) for x in args])
        print(f'[{i}] {line}', flush=True)
        if not a.dry_run:
            subprocess.run([py] + args, cwd=REPO, env={**os.environ, **env}, check=True)


if __name__ == '__main__':
    sys.exit(main())
