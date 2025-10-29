import os
import subprocess

GRAPHLAND_DATA_ROOT = '../datasets/graphland'
model_name = 'pretrn_gen1'

for dataset_name in sorted(os.listdir(GRAPHLAND_DATA_ROOT)):
    subprocess.call([
        'python', 'main.py',
        '--data_dir', GRAPHLAND_DATA_ROOT,
        '--load', model_name,
        '--epoch', '0',
        '--tstdata', dataset_name
    ])
