import os
import subprocess

GRAPHLAND_DATA_ROOT = '../datasets/graphland'
model_name = 'pretrain_link1'

for dataset_name in sorted(os.listdir(GRAPHLAND_DATA_ROOT)):
    subprocess.call([
        'python', 'main.py',
        '--load', model_name,
        '--epoch', '0',
        '--dataset', dataset_name
    ])
