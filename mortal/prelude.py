import sys
import os
import logging
import warnings
import torch
import numpy as np

sys.stdin.reconfigure(encoding='utf-8')

# Ensure repo root is on path so libne is importable from anywhere
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

logging.basicConfig(
    stream = sys.stderr,
    level = logging.INFO,
    format = '%(asctime)s %(levelname)8s %(filename)12s:%(lineno)-4s %(message)s',
)

_file_handler = logging.FileHandler('train.log', encoding='utf-8')
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)8s %(filename)12s:%(lineno)-4s %(message)s'))
logging.getLogger().addHandler(_file_handler)

warnings.simplefilter('ignore')

# "The given NumPy array is not writeable"
dummy = np.array([])
dummy.setflags(write=False)
torch.as_tensor(dummy)

# "distutils Version classes are deprecated"
import torch.utils.tensorboard

warnings.simplefilter('default')
