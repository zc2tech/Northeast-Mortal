"""
obs_encoder.py — encode NortheastPlayerState into a float32 observation array.

Layout (OBS_ROWS × 34):
  Row range   Count  Feature
  ---------   -----  -------
    0-3           4  Tehai counts (row k = 1 if tehai[tile] > k)
    4-9           6  Self kawa first 6 items (1 row each)
    10-27        18  Self kawa last 18 items (1 row each)
    28            1  Self kawa recency
    29-34         6  Opponent +1 kawa first 6 (1 row each)
    35-52        18  Opponent +1 kawa last 18 (1 row each)
    53            1  Opponent +1 kawa recency
    54-59         6  Opponent +2 kawa first 6 (1 row each)
    60-77        18  Opponent +2 kawa last 18 (1 row each)
    78            1  Opponent +2 kawa recency
    79-84         6  Opponent +3 kawa first 6 (1 row each)
    85-102       18  Opponent +3 kawa last 18 (1 row each)
    103           1  Opponent +3 kawa recency
    104           1  Tiles left /69
    105-108       4  Kawa overview self
    109-112       4  Kawa overview p+1
    113-116       4  Kawa overview p+2
    117-120       4  Kawa overview p+3
    121-136      16  Fuuro self  (4 melds × 4 rows)
    137-152      16  Fuuro p+1
    153-168      16  Fuuro p+2
    169-184      16  Fuuro p+3
    185-188       4  Ankan overview (1 row per seat)
    189           1  Tiles seen /4
    190           1  Last discard p+1
    191           1  Last discard p+2
    192           1  Last discard p+3
    193           1  Weighted shanten scalar (shanten/10, -1/10=win)
    194           1  Last kawa tile
    195-204      10  Action availability
    205           1  Dead wall marker (count/4 per tile; no dora meaning in Northeast rules)
    206           1  Jikaze (seat wind): col 27=E,28=S,29=W,30=N set to 1.0
    207           1  Bakaze (round wind): col 27=E,28=S,29=W,30=N set to 1.0
    208-212       5  Post-call weighted_shanten/10: [chi_low, chi_mid, chi_high, pon, kan] (-1=unavailable)
    213           1  Has formed sequence with terminal tile (in hand melds or open melds)
    214           1  Has formed triplet/kan with terminal or dragon tile (open or ankan)
    215           1  Has terminal tile in hand or open melds

Total: 216 rows (OBS_ROWS = 216)
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .player_state import NortheastPlayerState

TILE_COUNT = 34
_OBS_COLS = 34

# When False, weighted_shanten and post_call_shanten rows are omitted (saves 6 rows).
# Must match the flag used when the model was trained.
USE_SHANTEN_FEATURES = True

def _apply_obs_config() -> None:
    try:
        import os as _os, toml as _toml
        _cfg_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            'mortal', 'config.toml',
        )
        if not _os.path.exists(_cfg_path):
            return
        _cfg = _toml.load(_cfg_path)
        global USE_SHANTEN_FEATURES
        USE_SHANTEN_FEATURES = _cfg.get('obs', {}).get('use_shanten_features', True)
    except Exception:
        pass

_apply_obs_config()

_OBS_ROWS_WITH_SHANTEN    = 216
_OBS_ROWS_WITHOUT_SHANTEN = 210

def _obs_rows() -> int:
    return _OBS_ROWS_WITH_SHANTEN if USE_SHANTEN_FEATURES else _OBS_ROWS_WITHOUT_SHANTEN


def _assign_row(arr: np.ndarray, row: int, col: int, val: float) -> None:
    if 0 <= row < arr.shape[0] and 0 <= col < _OBS_COLS:
        arr[row, col] = val


def _fill_row(arr: np.ndarray, row: int, val: float) -> None:
    if 0 <= row < arr.shape[0]:
        arr[row, :] = val


def _encode_tile_set(arr: np.ndarray, base_row: int,
                     tile_counts: list[int]) -> None:
    """Encode a tile count set: 4 count rows."""
    for tile_idx in range(TILE_COUNT):
        cnt = tile_counts[tile_idx]
        for k in range(min(cnt, 4)):
            _assign_row(arr, base_row + k, tile_idx, 1.0)


def encode_obs(state: "NortheastPlayerState") -> np.ndarray:
    arr = np.zeros((_obs_rows(), _OBS_COLS), dtype=np.float32)
    row = 0

    # ── Tehai counts (4 rows) ──
    for k in range(4):
        for tile_idx in range(TILE_COUNT):
            if state.tehai[tile_idx] > k:
                arr[row + k, tile_idx] = 1.0
    row += 4

    # ── Self kawa (first 6 × 1 row, then last 18 × 1 row, then recency) ──
    row = _encode_self_kawa(arr, row, state)

    # ── Opponent kawas × 3 (first 6 × 1 row, last 18 × 1 row, recency) ──
    for opp_offset in range(1, 4):
        opp = (state.player_id + opp_offset) % 4
        row = _encode_opp_kawa(arr, row, state, opp)

    # ── Tiles left (1 row) ──
    arr[row, :] = state.tiles_left / 69.0
    row += 1

    # ── Kawa overview × 4 (4 rows each) ──
    for i in range(4):
        seat = (state.player_id + i) % 4
        kawa_counts = [0] * TILE_COUNT
        for item in state.kawa[seat]:
            if 0 <= item.tile < TILE_COUNT:
                kawa_counts[item.tile] += 1
        _encode_tile_set(arr, row, kawa_counts)
        row += 4

    # ── Fuuro overview × 4 (4 melds × 5 rows = 20 rows each) ──
    for i in range(4):
        seat = (state.player_id + i) % 4
        row = _encode_fuuro(arr, row, state.fuuro[seat] + [
            [state.ankan[seat][j]] * 4 for j in range(len(state.ankan[seat]))
        ])

    # ── Ankan overview × 4 (1 row each) ──
    for i in range(4):
        seat = (state.player_id + i) % 4
        for t in state.ankan[seat]:
            _assign_row(arr, row, t, 1.0)
        row += 1

    # ── Tiles seen (1 row) ──
    tiles_seen = [0] * TILE_COUNT
    for seat in range(4):
        for item in state.kawa[seat]:
            if 0 <= item.tile < TILE_COUNT:
                tiles_seen[item.tile] += 1
        for meld in state.fuuro[seat]:
            for t in meld:
                if 0 <= t < TILE_COUNT:
                    tiles_seen[t] += 1
    for tile_idx in range(TILE_COUNT):
        arr[row, tile_idx] = tiles_seen[tile_idx] / 4.0
    row += 1

    # ── Last discard × 3 opponents (1 row each) ──
    for opp_offset in range(1, 4):
        opp = (state.player_id + opp_offset) % 4
        kawa = state.kawa[opp]
        if kawa:
            _assign_row(arr, row, kawa[-1].tile, 1.0)
        row += 1

    # ── Weighted shanten scalar (1 row) ──
    if USE_SHANTEN_FEATURES:
        sht = max(-1, min(10, state.shanten))
        arr[row, :] = sht / 10.0
        row += 1

    # ── Last kawa tile (1 row) ──
    if 0 <= state.last_kawa_tile < TILE_COUNT:
        _assign_row(arr, row, state.last_kawa_tile, 1.0)
    row += 1

    # ── Action availability (10 rows) ──
    if state._can_discard:
        arr[row, :] = 1.0
    row += 1
    if state._can_chi_low:
        arr[row, :] = 1.0
    row += 1
    if state._can_chi_mid:
        arr[row, :] = 1.0
    row += 1
    if state._can_chi_high:
        arr[row, :] = 1.0
    row += 1
    if state._can_pon:
        arr[row, :] = 1.0
    row += 1
    if state._can_kan:
        arr[row, :] = 1.0
    row += 1
    if state._can_agari:
        arr[row, :] = 1.0
    row += 1
    if state._can_ryukyoku:
        arr[row, :] = 1.0
    row += 1
    row += 2  # pass + spare

    # ── Dead wall marker (1 row) ──
    for idx in state.dead_wall_markers:
        if 0 <= idx < TILE_COUNT:
            arr[row, idx] = min(arr[row, idx] + 0.25, 1.0)
    row += 1

    # ── Jikaze / Bakaze (1 row each) ──
    _assign_row(arr, row, state.jikaze, 1.0)
    row += 1
    _assign_row(arr, row, state.bakaze, 1.0)
    row += 1

    # ── Post-call weighted_shanten (5 rows: chi_low/mid/high/pon/kan) ──
    if USE_SHANTEN_FEATURES:
        for k, sht in enumerate(state.post_call_shanten):
            arr[row + k, :] = sht / 10.0 if sht >= 0 else -1.0
        row += 5

    # ── Terminal-sequence feature (1 row) ──
    _TERMINAL_SET = frozenset([0, 8, 9, 17, 18, 26])
    has_terminal_seq = False
    for meld in state.fuuro[state.player_id]:
        if len(meld) >= 3:
            a, b, c = sorted(meld[:3])
            if a < 27 and (a // 9 == b // 9 == c // 9) and b == a + 1 and c == a + 2:
                if a in _TERMINAL_SET or c in _TERMINAL_SET:
                    has_terminal_seq = True
                    break
    if not has_terminal_seq:
        for suit in range(3):
            base = suit * 9
            for start in range(base, base + 7):
                if state.tehai[start] > 0 and state.tehai[start + 1] > 0 and state.tehai[start + 2] > 0:
                    if start in _TERMINAL_SET or (start + 2) in _TERMINAL_SET:
                        has_terminal_seq = True
                        break
            if has_terminal_seq:
                break
    if has_terminal_seq:
        arr[row, :] = 1.0
    row += 1

    # ── Terminal/dragon triplet feature (1 row) ──
    _YAOCHUU_SET = frozenset([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33])
    has_yaochuu_triplet = False
    for meld in state.fuuro[state.player_id]:
        if len(meld) >= 3 and len(set(meld[:3])) == 1:
            if meld[0] in _YAOCHUU_SET:
                has_yaochuu_triplet = True
                break
    if not has_yaochuu_triplet:
        for t in state.ankan[state.player_id]:
            if t in _YAOCHUU_SET:
                has_yaochuu_triplet = True
                break
    if not has_yaochuu_triplet:
        for idx in _YAOCHUU_SET:
            if state.tehai[idx] >= 3:
                has_yaochuu_triplet = True
                break
    if has_yaochuu_triplet:
        arr[row, :] = 1.0
    row += 1

    # ── Terminal tile present feature (1 row) ──
    has_terminal = False
    for idx in _TERMINAL_SET:
        if state.tehai[idx] > 0:
            has_terminal = True
            break
    if not has_terminal:
        for meld in state.fuuro[state.player_id]:
            for t in meld:
                if t in _TERMINAL_SET:
                    has_terminal = True
                    break
            if has_terminal:
                break
    if not has_terminal:
        for t in state.ankan[state.player_id]:
            if t in _TERMINAL_SET:
                has_terminal = True
                break
    if has_terminal:
        arr[row, :] = 1.0
    row += 1

    assert row == _obs_rows(), f"Row count mismatch: {row} != {_obs_rows()}"
    return arr


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _encode_self_kawa(arr: np.ndarray, row: int, state: "NortheastPlayerState") -> int:
    import math
    kawa = state.kawa[state.player_id]
    first6 = kawa[:6] if len(kawa) >= 6 else kawa + [None] * (6 - len(kawa))
    for item in first6:
        if item is not None:
            _assign_row(arr, row, item.tile, 1.0)
        row += 1

    last18_src = kawa[6:24] if len(kawa) > 6 else []
    last18 = last18_src + [None] * (18 - len(last18_src))
    for item in last18:
        if item is not None:
            _assign_row(arr, row, item.tile, 1.0)
        row += 1

    # Recency map (1 row) — exponential decay over kawa
    for i, item in enumerate(kawa):
        if item is not None and 0 <= item.tile < TILE_COUNT:
            w = math.exp(-0.1 * (len(kawa) - 1 - i))
            arr[row, item.tile] = max(arr[row, item.tile], w)
    row += 1
    return row


def _encode_opp_kawa(arr: np.ndarray, row: int, state: "NortheastPlayerState", opp: int) -> int:
    import math
    kawa = state.kawa[opp]
    first6 = kawa[:6] if len(kawa) >= 6 else kawa + [None] * (6 - len(kawa))
    for item in first6:
        if item is not None:
            _assign_row(arr, row, item.tile, 1.0)
        row += 1

    last18_src = kawa[6:24] if len(kawa) > 6 else []
    last18 = last18_src + [None] * (18 - len(last18_src))
    for item in last18:
        if item is not None:
            _assign_row(arr, row, item.tile, 1.0)
        row += 1

    # Recency map (1 row)
    for i, item in enumerate(kawa):
        if item is not None and 0 <= item.tile < TILE_COUNT:
            w = math.exp(-0.1 * (len(kawa) - 1 - i))
            arr[row, item.tile] = max(arr[row, item.tile], w)
    row += 1
    return row


def _encode_fuuro(arr: np.ndarray, row: int, melds: list[list[int]]) -> int:
    """Encode up to 4 melds × 4 rows (tile-count encoding)."""
    for meld_idx in range(4):
        if meld_idx < len(melds):
            meld = melds[meld_idx]
            counts = [0] * TILE_COUNT
            for t in meld:
                if 0 <= t < TILE_COUNT:
                    counts[t] += 1
            for t in range(TILE_COUNT):
                for k in range(min(counts[t], 4)):
                    arr[row + k, t] = 1.0
        row += 4
    return row
