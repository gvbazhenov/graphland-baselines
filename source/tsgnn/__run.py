import subprocess

TRAIN_SIZE = 9
N_HIDDEN_CHANNELS = 32
N_LAYERS = 3
IS_TRAIN = False

commands = [
    'python', 'main.py',
    '--project', 'graphland',
    '--train_test_setup', 'inc_trainset',
    '--train_size', str(TRAIN_SIZE),
    '--hid_channel', str(N_HIDDEN_CHANNELS),
    '--num_layers', str(N_LAYERS),
    '--is_train'
]

if IS_TRAIN:
    subprocess.call(commands)

commands.pop()
subprocess.call(commands)
