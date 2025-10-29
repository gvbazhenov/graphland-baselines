import os
from torch.nn import CrossEntropyLoss

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECIMAL = 4
SEEDS = list(range(5))
TASK_LOSS = CrossEntropyLoss()

API_TOKEN = "..."  # insert your token to use neptune
