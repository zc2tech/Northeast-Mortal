"""
Shanten calculator ported from Northeast-Mahjong/source/GameServer/Bots/ShantenCalculator.vala.
Uses the same precomputed lookup tables (index_h.bin, index_s.bin) from tomohxx/shanten-number.

Returns shanten number:
  -1 = winning hand (complete)
   0 = tenpai
   1 = 1-shanten
   etc.

The Vala code returns shanten+1 in its public API (0=winning, 1=tenpai, ...).
This Python port returns standard shanten values (-1=winning, 0=tenpai, ...) by subtracting 1.
"""

import os
import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

_index_h: np.ndarray | None = None  # shape (78125, 10) uint8
_index_s: np.ndarray | None = None  # shape (1953125, 10) uint8


def _load_tables() -> None:
    global _index_h, _index_s
    h_path = os.path.join(_DATA_DIR, "index_h.bin")
    s_path = os.path.join(_DATA_DIR, "index_s.bin")
    _index_h = np.fromfile(h_path, dtype=np.uint8).reshape(-1, 10)
    _index_s = np.fromfile(s_path, dtype=np.uint8).reshape(-1, 10)


def _ensure_loaded() -> None:
    if _index_h is None:
        _load_tables()


def _hash9(tiles: list[int], offset: int) -> int:
    """Base-5 hash for 9 suited tiles."""
    h = 0
    m = 1
    for i in range(9):
        h += tiles[offset + i] * m
        m *= 5
    return h


def _hash7(tiles: list[int], offset: int) -> int:
    """Base-5 hash for 7 honor tiles."""
    h = 0
    m = 1
    for i in range(7):
        h += tiles[offset + i] * m
        m *= 5
    return h


def _add1(lhs: list[int], rhs: list[int], m: int) -> None:
    """Combine accumulated shanten results (lhs) with a new suit (rhs) in-place."""
    # Has-pair states (indices 5..m+5)
    for j in range(m + 5, 4, -1):
        sht = min(lhs[j] + rhs[0], lhs[0] + rhs[j])
        for k in range(5, j):
            sht = min(sht, lhs[k] + rhs[j - k], lhs[j - k] + rhs[k])
        lhs[j] = sht

    # No-pair states (indices 0..m)
    for j in range(m, -1, -1):
        sht = lhs[j] + rhs[0]
        for k in range(j):
            sht = min(sht, lhs[k] + rhs[j - k])
        lhs[j] = sht


def _add2(lhs: list[int], rhs: list[int], m: int) -> None:
    """Final-suit combination: only compute index m+5."""
    j = m + 5
    sht = min(lhs[j] + rhs[0], lhs[0] + rhs[j])
    for k in range(5, j):
        sht = min(sht, lhs[k] + rhs[j - k], lhs[j - k] + rhs[k])
    lhs[j] = sht


def _calc_lh(tiles: list[int], m: int) -> int:
    """Core shanten calculation for standard form (4 melds + 1 pair)."""
    assert _index_h is not None and _index_s is not None

    h_honors = _hash7(tiles, 27)
    ret = list(map(int, _index_h[h_honors]))

    h_sou = _hash9(tiles, 18)
    sou = list(map(int, _index_s[h_sou]))
    _add1(ret, sou, m)

    h_pin = _hash9(tiles, 9)
    pin = list(map(int, _index_s[h_pin]))
    _add1(ret, pin, m)

    h_man = _hash9(tiles, 0)
    man = list(map(int, _index_s[h_man]))
    _add2(ret, man, m)

    # ret[m+5] = shanten+1 in the Vala convention (0=winning, 1=tenpai)
    # Subtract 1 to get standard shanten (-1=winning, 0=tenpai)
    return ret[m + 5] - 1


def calc_shanten(tile_counts: list[int], open_meld_count: int = 0) -> int:
    """
    Calculate shanten number for a closed hand.

    Args:
        tile_counts: list of 34 ints, counts per tile index 0-33
        open_meld_count: number of open melds (pon/chi/kan already declared)

    Returns:
        shanten: -1=winning, 0=tenpai, 1=1-shanten, ...
    """
    _ensure_loaded()

    closed_tiles = sum(tile_counts)
    m = closed_tiles // 3

    return _calc_lh(tile_counts, m)


_shanten_cache: dict = {}
_CACHE_MAX = 65536

def calc_shanten_cached(tile_counts: list[int], open_meld_count: int = 0) -> int:
    key = (tuple(tile_counts), open_meld_count)
    result = _shanten_cache.get(key)
    if result is not None:
        return result
    result = calc_shanten(tile_counts, open_meld_count)
    if len(_shanten_cache) >= _CACHE_MAX:
        _shanten_cache.clear()
    _shanten_cache[key] = result
    return result
