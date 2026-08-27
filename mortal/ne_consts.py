"""
Shim providing libriichi.consts-compatible API for Northeast Mahjong training.
obs_shape(version) accepts any version; always returns (OBS_ROWS, OBS_COLS).
"""

import sys
import os

# Ensure the Northeast-Mortal root (parent of mortal/) is on the path so libne is importable
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from libne.consts import ACTION_SPACE, OBS_COLS
import libne.obs_encoder as _enc

GRP_SIZE = 7


def _apply_config() -> None:
    try:
        from config import config
        use_shanten = config.get('obs', {}).get('use_shanten_features', True)
        _enc.USE_SHANTEN_FEATURES = use_shanten
    except Exception:
        pass

_apply_config()


def obs_shape(version: int = 1) -> tuple[int, int]:
    return (_enc._obs_rows(), OBS_COLS)


def oracle_obs_shape(version: int = 1) -> tuple[int, int]:
    return (0, OBS_COLS)
