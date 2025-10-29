import os
import subprocess

EXPERIMENT_ROOT = 'exp/lightgbm_'

for experiment_name in sorted(os.listdir(EXPERIMENT_ROOT)):
    config_path = f'exp/lightgbm_/{experiment_name}/test-tuning.toml'
    subprocess.call(['python', 'bin/go.py', config_path, '--force'])
