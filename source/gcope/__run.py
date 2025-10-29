import os
import subprocess

GRAPHLAND_DATA_ROOT = './datasets/graphland'

pretrain_dataset_names = [
    'wisconsin',
    'texas',
    'cornell',
    # 'chameleon',
    # 'squirrel',
    'cora',
    'citeseer',
    'pubmed',
    'computers',
    'photo'
]

hidden_dim = 128
num_epochs = 100
batch_size = 256
backbone_name = 'fagcn'
learning_rate = 0.001

# >>> pretrain

pretrain_dataset_str = ','.join(pretrain_dataset_names)
storage_path = './storage'

checkpoint_path = f'{storage_path}/{pretrain_dataset_str}_pretrained_model.pt'
if not os.path.exists(checkpoint_path):
    subprocess.call([
        'python', 'src/exec.py',
        '--config-file', 'pretrain.json',
        '--general.save_dir', storage_path,
        '--general.reconstruct', str(0.2),
        '--data.node_feature_dim', str(hidden_dim),
        '--data.name', pretrain_dataset_str,
        '--pretrain.split_method', 'RandomWalk',
        '--model.backbone.model_type', backbone_name,
        '--pretrain.epoch', str(num_epochs),
    ])

# >>> finetune

for dataset_name in os.listdir(GRAPHLAND_DATA_ROOT):
    experiment_path = f'{storage_path}/{dataset_name}'
    subprocess.call([
        'python', 'src/exec.py',
        '--general.func', 'adapt',
        '--general.save_dir', experiment_path,
        '--general.few_shot', '0',
        '--general.reconstruct', '0.0',
        '--data.node_feature_dim', str(hidden_dim),
        '--data.name', dataset_name,
        '--adapt.method', 'finetune',
        '--model.backbone.model_type', backbone_name,
        '--model.saliency.model_type', 'none',
        '--adapt.pretrained_file', checkpoint_path,
        '--adapt.finetune.learning_rate', str(learning_rate),
        '--adapt.batch_size', str(batch_size),
        '--adapt.finetune.backbone_tuning', '1',
        '--adapt.epoch', str(num_epochs),
    ])
