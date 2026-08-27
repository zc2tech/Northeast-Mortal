"""mjai tile string ↔ tile index (0-33) conversion."""

# mjai string → tile index (0-33)
_MJAI_TO_IDX = {}

# Man 1m-9m → 0-8
for _i, _s in enumerate(["1m","2m","3m","4m","5m","6m","7m","8m","9m"]):
    _MJAI_TO_IDX[_s] = _i

# Pin 1p-9p → 9-17
for _i, _s in enumerate(["1p","2p","3p","4p","5p","6p","7p","8p","9p"]):
    _MJAI_TO_IDX[_s] = 9 + _i

# Sou 1s-9s → 18-26
for _i, _s in enumerate(["1s","2s","3s","4s","5s","6s","7s","8s","9s"]):
    _MJAI_TO_IDX[_s] = 18 + _i

# Honors → 27-33
_MJAI_TO_IDX.update({
    "E": 27, "S": 28, "W": 29, "N": 30,
    "P": 31, "F": 32, "C": 33,
})

# Reverse mapping
_IDX_TO_MJAI = {v: k for k, v in _MJAI_TO_IDX.items()}


def mjai_to_idx(tile_str: str) -> int:
    """Convert mjai tile string to 0-33 index. Returns -1 for unknown."""
    return _MJAI_TO_IDX.get(tile_str, -1)


def idx_to_mjai(idx: int) -> str:
    """Convert tile index 0-33 to mjai string."""
    return _IDX_TO_MJAI.get(idx, "?")
