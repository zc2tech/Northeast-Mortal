"""
NortheastPlayerState: tracks game state by consuming mjai events.
One instance per player seat. Produces (obs_snapshot, action_label, mask) at each decision point.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Optional

from .consts import (
    ACTION_SPACE, TILE_COUNT,
    ACTION_DISCARD_BASE, ACTION_CHI_LOW, ACTION_CHI_MID, ACTION_CHI_HIGH,
    ACTION_PON, ACTION_KAN, ACTION_AGARI, ACTION_RYUKYOKU, ACTION_PASS,
    WIND_INDEX,
)
from .tile import mjai_to_idx


class KawaItem:
    """One item in a player's discard river."""
    __slots__ = ("tile", "from_kan")

    def __init__(self, tile: int, from_kan: bool = False):
        self.tile = tile
        self.from_kan = from_kan


class NortheastPlayerState:
    """
    Tracks full game state from the perspective of one player seat.
    Call update(event) for each mjai event in order.
    """

    def __init__(self, player_id: int):
        self.player_id = player_id
        self._reset()

    def _reset(self) -> None:
        # Preserve hook across resets (set by tests; not part of game state)
        _hook = getattr(self, "_obs_hook", None)
        # Hand tiles: count per tile index 0-33
        self.tehai = np.zeros(TILE_COUNT, dtype=np.int32)

        # Discard rivers: list of KawaItem per player (4 players)
        self.kawa: list[list[KawaItem]] = [[] for _ in range(4)]

        # Open melds: list of meld tile lists per player
        # Each meld = list of tile indices
        self.fuuro: list[list[list[int]]] = [[] for _ in range(4)]

        # Closed kans (ankans) per player: list of tile index
        self.ankan: list[list[int]] = [[] for _ in range(4)]

        # Scores (rotated so self=index 0)
        self.scores = [0, 0, 0, 0]

        # Wind indices (27-30)
        self.bakaze = WIND_INDEX["E"]
        self.jikaze = WIND_INDEX["E"]

        # Round info
        self.kyoku = 0    # 0-indexed
        self.honba = 0
        self.dealer = 0

        # Tiles remaining in wall
        self.tiles_left = 69

        # Dora indicators visible on the dead wall (cannot be drawn)
        self.dead_wall_markers: list[int] = []

        # Draw counter
        self.at_turn = 0

        # Shanten of own hand
        self.shanten = 8

        # Last tile discarded into the pond (for chi/pon/ron decisions)
        self.last_kawa_tile: int = -1
        self.last_kawa_player: int = -1

        # Last drawn tile (for tsumogiri detection)
        self.last_tsumo: Optional[int] = None

        # Action candidates for current decision point
        self._can_discard = False
        self._can_chi_low = False
        self._can_chi_mid = False
        self._can_chi_high = False
        self._can_pon = False
        self._can_kan = False
        self._can_agari = False
        self._can_ryukyoku = False

        # Pending post-draw decision: (obs, mask) snapshot taken after tsumo,
        # resolved when the following dahai/hora/kan event arrives.
        self._pending_decision: Optional[tuple[np.ndarray, np.ndarray]] = None

        # Pending call decision: (obs, mask) snapshot taken after an opponent
        # discards a tile that this player *could* call. Resolved when the next
        # event reveals what actually happened (call or pass).
        self._pending_call_decision: Optional[tuple[np.ndarray, np.ndarray]] = None

        # Hypothetical weighted_shanten after each possible call (chi_low/mid/high/pon/kan).
        # -1 means the call is not available. Updated in _snapshot_call_decision.
        self.post_call_shanten: list[int] = [-1, -1, -1, -1, -1]  # [chi_low, chi_mid, chi_high, pon, kan]

        # Optional callback fired in _encode_obs_snapshot(state, obs) — used for testing only.
        self._obs_hook = _hook

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, event: dict) -> Optional[tuple[np.ndarray, int, np.ndarray]]:
        """
        Process one mjai event.

        Returns (obs, action_label, mask) if this event represents a decision
        made by this player that we should train on, otherwise None.
        """
        t = event.get("type", "")

        if t == "start_kyoku":
            self._on_start_kyoku(event)
        elif t == "tsumo":
            return self._on_tsumo(event)
        elif t == "dahai":
            return self._on_dahai(event)
        elif t == "chi":
            return self._on_chi(event)
        elif t == "pon":
            return self._on_pon(event)
        elif t == "daiminkan":
            return self._on_daiminkan(event)
        elif t == "kakan":
            return self._on_kakan(event)
        elif t == "ankan":
            return self._on_ankan(event)
        elif t == "hora":
            return self._on_hora(event)
        elif t == "ryukyoku":
            return self._on_ryukyoku(event)
        return None

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_start_kyoku(self, ev: dict) -> None:
        self._reset()
        self.bakaze = WIND_INDEX.get(ev.get("bakaze", "E"), WIND_INDEX["E"])
        self.kyoku = ev.get("kyoku", 1) - 1   # store 0-indexed
        self.honba = ev.get("honba", 0)
        self.dealer = ev.get("oya", 0)

        raw_scores = ev.get("scores", [0, 0, 0, 0])
        # Rotate so self=index 0
        for i in range(4):
            self.scores[i] = raw_scores[(self.player_id + i) % 4]

        # Seat wind: dealer=East, +1 per seat
        seat_wind_names = ["E", "S", "W", "N"]
        seat_offset = (self.player_id - self.dealer) % 4
        self.jikaze = WIND_INDEX[seat_wind_names[seat_offset]]

        # Initial dead wall marker
        dwm_str = ev.get("dead_wall_marker", "")
        if dwm_str:
            idx = mjai_to_idx(dwm_str)
            if idx >= 0:
                self.dead_wall_markers.append(idx)

        tehais = ev.get("tehais", [])
        if self.player_id < len(tehais):
            for t_str in tehais[self.player_id]:
                idx = mjai_to_idx(t_str)
                if idx >= 0:
                    self.tehai[idx] += 1

        self._update_shanten()

    def _on_tsumo(self, ev: dict) -> Optional[tuple[np.ndarray, int, np.ndarray]]:
        actor = ev.get("actor", -1)
        tile_str = ev.get("pai", "")
        tile_idx = mjai_to_idx(tile_str)
        self.tiles_left = max(0, self.tiles_left - 1)
        self.at_turn += 1
        self.last_tsumo = tile_idx

        # A tsumo by anyone resolves any pending call decision for this player
        # as a pass (they chose not to call the previous discard).
        pass_sample = None
        if self._pending_call_decision is not None:
            obs_snap, mask = self._pending_call_decision
            self._pending_call_decision = None
            # Only emit pass sample if this player actually had a callable option
            if mask[ACTION_CHI_LOW] or mask[ACTION_CHI_MID] or mask[ACTION_CHI_HIGH] \
                    or mask[ACTION_PON] or mask[ACTION_KAN] or mask[ACTION_AGARI]:
                pass_sample = (obs_snap, ACTION_PASS, mask)

        if actor == self.player_id:
            if tile_idx >= 0:
                self.tehai[tile_idx] += 1
            self._update_shanten()
            # After drawing, player must discard (or kan/agari)
            self._can_discard = True
            self._can_agari = False  # tsumo win handled via TsumoClientAction → hora
            self._can_ryukyoku = False
            # Store snapshot for when we see the dahai/hora/kan that follows
            self._pending_decision = self._snapshot_decision()

        return pass_sample

    def _on_dahai(self, ev: dict) -> Optional[tuple[np.ndarray, int, np.ndarray]]:
        actor = ev.get("actor", -1)
        tile_str = ev.get("pai", "")
        tile_idx = mjai_to_idx(tile_str)

        if actor == self.player_id and self._pending_decision is not None:
            # Emit training sample: action = discard tile_idx
            action_label = tile_idx if tile_idx >= 0 else 0
            obs_snap, mask = self._pending_decision
            self._pending_decision = None
            # Execute discard
            self._do_discard(actor, tile_idx)
            return (obs_snap, action_label, mask)

        # Other player discarded
        self._do_discard(actor, tile_idx)

        # After an opponent discards, snapshot the call decision state for this
        # player. Resolved in _on_tsumo (pass) or _on_chi/_on_pon/etc. (call).
        if actor != self.player_id:
            self._snapshot_call_decision(tile_idx, actor)

        return None

    def _do_discard(self, actor: int, tile_idx: int) -> None:
        if actor == self.player_id and tile_idx >= 0:
            if self.tehai[tile_idx] > 0:
                self.tehai[tile_idx] -= 1
            self._update_shanten()
        self.kawa[actor].append(KawaItem(tile_idx))
        self.last_kawa_tile = tile_idx
        self.last_kawa_player = actor
        self.last_tsumo = None

    def _snapshot_call_decision(self, tile_idx: int, discard_actor: int) -> None:
        """
        After an opponent discards tile_idx, determine which calls this player
        could make, set the _can_* flags accordingly, and snapshot for training.
        If no calls are possible, no snapshot is stored.
        """
        self._can_discard = False
        self._can_chi_low = False
        self._can_chi_mid = False
        self._can_chi_high = False
        self._can_pon = False
        self._can_kan = False
        self._can_agari = False
        self._can_ryukyoku = False

        if tile_idx < 0:
            return

        # Chi: only the player directly to the left of the discarder (downstream)
        downstream = (discard_actor + 1) % 4
        if self.player_id == downstream:
            chi_low, chi_mid, chi_high = self._check_chi_options(tile_idx)
            self._can_chi_low = chi_low
            self._can_chi_mid = chi_mid
            self._can_chi_high = chi_high

        # Pon: need 2+ copies in hand
        if self.tehai[tile_idx] >= 2:
            self._can_pon = True

        # Open kan: need 3 copies in hand
        if self.tehai[tile_idx] >= 3:
            self._can_kan = True

        # Agari (ron): check if this tile completes the hand
        if self._check_ron(tile_idx):
            self._can_agari = True

        has_any_call = (self._can_chi_low or self._can_chi_mid or self._can_chi_high
                        or self._can_pon or self._can_kan or self._can_agari)
        if has_any_call:
            # Compute hypothetical post-call shantens for each available call
            from . import obs_encoder as _enc
            if _enc.USE_SHANTEN_FEATURES:
                self.post_call_shanten = [
                    self._calc_post_call_shanten(tile_idx, 'chi_low')  if self._can_chi_low  else -1,
                    self._calc_post_call_shanten(tile_idx, 'chi_mid')  if self._can_chi_mid  else -1,
                    self._calc_post_call_shanten(tile_idx, 'chi_high') if self._can_chi_high else -1,
                    self._calc_post_call_shanten(tile_idx, 'pon')      if self._can_pon      else -1,
                    self._calc_post_call_shanten(tile_idx, 'kan')      if self._can_kan      else -1,
                ]
            else:
                self.post_call_shanten = [-1, -1, -1, -1, -1]
            # Also allow pass as an explicit option
            mask = self._make_call_mask()
            obs_snap = self._encode_obs_snapshot()
            self._pending_call_decision = (obs_snap, mask)

        # Reset flags — they'll be re-set properly when the call event arrives
        self._can_chi_low = False
        self._can_chi_mid = False
        self._can_chi_high = False
        self._can_pon = False
        self._can_kan = False
        self._can_agari = False
        if not has_any_call:
            self.post_call_shanten = [-1, -1, -1, -1, -1]

    def _make_call_mask(self) -> np.ndarray:
        """Build mask for a call-or-pass decision."""
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        if self._can_chi_low:
            mask[ACTION_CHI_LOW] = True
        if self._can_chi_mid:
            mask[ACTION_CHI_MID] = True
        if self._can_chi_high:
            mask[ACTION_CHI_HIGH] = True
        if self._can_pon:
            mask[ACTION_PON] = True
        if self._can_kan:
            mask[ACTION_KAN] = True
        if self._can_agari:
            mask[ACTION_AGARI] = True
        mask[ACTION_PASS] = True
        return mask

    def _check_chi_options(self, tile_idx: int) -> tuple[bool, bool, bool]:
        """
        Return (can_chi_low, can_chi_mid, can_chi_high) for the given tile.
        Chi is only valid for numbered tiles (0-26, i.e. man/pin/sou).
        low  = tile is the lowest  → need tile+1, tile+2 in hand
        mid  = tile is the middle  → need tile-1, tile+1 in hand
        high = tile is the highest → need tile-2, tile-1 in hand
        """
        if tile_idx >= 27:  # honour tile, no chi
            return False, False, False
        suit = tile_idx // 9
        num = tile_idx % 9  # 0-indexed within suit

        def has(t: int) -> bool:
            return 0 <= t < TILE_COUNT and (t // 9) == suit and self.tehai[t] > 0

        chi_low  = (num <= 6) and has(tile_idx + 1) and has(tile_idx + 2)
        chi_mid  = (1 <= num <= 7) and has(tile_idx - 1) and has(tile_idx + 1)
        chi_high = (num >= 2) and has(tile_idx - 2) and has(tile_idx - 1)
        return chi_low, chi_mid, chi_high

    def _check_ron(self, tile_idx: int) -> bool:
        return False

    def _emit_pass_if_pending(self) -> Optional[tuple[np.ndarray, int, np.ndarray]]:
        """If we had a callable opportunity that another player claimed, emit a pass sample."""
        if self._pending_call_decision is not None:
            obs_snap, mask = self._pending_call_decision
            self._pending_call_decision = None
            return (obs_snap, ACTION_PASS, mask)
        return None

    def _on_chi(self, ev: dict) -> Optional[tuple[np.ndarray, int, np.ndarray]]:
        actor = ev.get("actor", -1)
        pai_str = ev.get("pai", "")
        consumed = [mjai_to_idx(s) for s in ev.get("consumed", [])]
        called_idx = mjai_to_idx(pai_str)

        # Remove called tile from kawa of target
        target = ev.get("target", self.last_kawa_player)
        if self.kawa[target]:
            self.kawa[target].pop()

        if actor == self.player_id:
            # Determine chi variant (low/mid/high) from consumed + called tile
            all_tiles = sorted([called_idx] + consumed)
            action_label = self._chi_action_label(all_tiles)

            # Use the pre-call snapshot if available, else build now
            if self._pending_call_decision is not None:
                obs_snap, mask = self._pending_call_decision
                self._pending_call_decision = None
            else:
                self._can_chi_low = self._can_chi_mid = self._can_chi_high = False
                self._can_pon = self._can_kan = self._can_agari = False
                obs_snap = self._encode_obs_snapshot()
                mask = self._make_mask()

            # Execute chi: remove consumed tiles from hand
            for t in consumed:
                if t >= 0 and self.tehai[t] > 0:
                    self.tehai[t] -= 1
            # Add the meld
            meld = sorted([called_idx] + consumed)
            self.fuuro[actor].append(meld)
            self._update_shanten()

            return (obs_snap, action_label, mask)

        else:
            # Other player called chi — remove called tile marker from kawa
            meld = sorted([called_idx] + consumed)
            self.fuuro[actor].append(meld)
            return self._emit_pass_if_pending()

    def _chi_action_label(self, sorted_tiles: list[int]) -> int:
        """Determine chi low/mid/high from the sorted 3-tile sequence."""
        from .consts import ACTION_CHI_LOW, ACTION_CHI_MID, ACTION_CHI_HIGH
        if len(sorted_tiles) < 3:
            return ACTION_CHI_LOW
        # called tile relative position in sequence
        called = self.last_kawa_tile
        if called == sorted_tiles[0]:
            return ACTION_CHI_LOW   # called is lowest → chi low (hand has mid+high)
        elif called == sorted_tiles[1]:
            return ACTION_CHI_MID
        else:
            return ACTION_CHI_HIGH  # called is highest → chi high (hand has low+mid)

    def _on_pon(self, ev: dict) -> Optional[tuple[np.ndarray, int, np.ndarray]]:
        actor = ev.get("actor", -1)
        pai_str = ev.get("pai", "")
        called_idx = mjai_to_idx(pai_str)
        consumed = [mjai_to_idx(s) for s in ev.get("consumed", [])]
        target = ev.get("target", self.last_kawa_player)

        if self.kawa[target]:
            self.kawa[target].pop()

        if actor == self.player_id:
            if self._pending_call_decision is not None:
                obs_snap, mask = self._pending_call_decision
                self._pending_call_decision = None
            else:
                obs_snap = self._encode_obs_snapshot()
                mask = self._make_mask()

            for t in consumed:
                if t >= 0 and self.tehai[t] > 0:
                    self.tehai[t] -= 1
            self.fuuro[actor].append([called_idx, called_idx, called_idx])
            self._update_shanten()

            return (obs_snap, ACTION_PON, mask)

        else:
            self.fuuro[actor].append([called_idx, called_idx, called_idx])
            return self._emit_pass_if_pending()

        return None

    def _on_daiminkan(self, ev: dict) -> Optional[tuple[np.ndarray, int, np.ndarray]]:
        actor = ev.get("actor", -1)
        pai_str = ev.get("pai", "")
        called_idx = mjai_to_idx(pai_str)
        consumed = [mjai_to_idx(s) for s in ev.get("consumed", [])]
        target = ev.get("target", self.last_kawa_player)

        if self.kawa[target]:
            self.kawa[target].pop()

        if actor == self.player_id:
            if self._pending_call_decision is not None:
                obs_snap, mask = self._pending_call_decision
                self._pending_call_decision = None
            else:
                obs_snap = self._encode_obs_snapshot()
                mask = self._make_mask()

            for t in consumed:
                if t >= 0 and self.tehai[t] > 0:
                    self.tehai[t] -= 1
            self.fuuro[actor].append([called_idx] * 4)
            self._update_shanten()

            return (obs_snap, ACTION_KAN, mask)

        else:
            self.fuuro[actor].append([called_idx] * 4)
            return self._emit_pass_if_pending()

        return None

    def _on_kakan(self, ev: dict) -> Optional[tuple[np.ndarray, int, np.ndarray]]:
        actor = ev.get("actor", -1)
        tile_str = ev.get("pai", "")
        tile_idx = mjai_to_idx(tile_str)

        if actor == self.player_id:
            obs_snap = self._encode_obs_snapshot()
            mask = self._make_mask()

            if tile_idx >= 0 and self.tehai[tile_idx] > 0:
                self.tehai[tile_idx] -= 1
            # Extend existing pon to kan
            for meld in self.fuuro[actor]:
                if len(meld) == 3 and meld[0] == tile_idx:
                    meld.append(tile_idx)
                    break
            self._update_shanten()
            self._pending_decision = None

            return (obs_snap, ACTION_KAN, mask)

        return None

    def _on_ankan(self, ev: dict) -> Optional[tuple[np.ndarray, int, np.ndarray]]:
        actor = ev.get("actor", -1)
        consumed = [mjai_to_idx(s) for s in ev.get("consumed", [])]
        tile_idx = consumed[0] if consumed else -1

        if actor == self.player_id:
            obs_snap = self._encode_obs_snapshot()
            mask = self._make_mask()

            # Remove 4 copies from hand
            if tile_idx >= 0:
                self.tehai[tile_idx] = max(0, self.tehai[tile_idx] - 4)
            self.ankan[actor].append(tile_idx)
            self._update_shanten()
            self._pending_decision = None

            return (obs_snap, ACTION_KAN, mask)

        else:
            if tile_idx >= 0:
                self.ankan[actor].append(tile_idx)

        return None

    def _on_hora(self, ev: dict) -> Optional[tuple[np.ndarray, int, np.ndarray]]:
        actor = ev.get("actor", -1)
        target = ev.get("target", actor)

        if actor == self.player_id:
            # Tsumo win: use pending post-draw snapshot if available
            if target == actor and self._pending_decision is not None:
                obs_snap, mask = self._pending_decision
                self._pending_decision = None
                mask = np.zeros(ACTION_SPACE, dtype=bool)
                mask[ACTION_AGARI] = True
                return (obs_snap, ACTION_AGARI, mask)

            # Ron win: use pending call snapshot if available
            if target != actor and self._pending_call_decision is not None:
                obs_snap, mask = self._pending_call_decision
                self._pending_call_decision = None
                mask = np.zeros(ACTION_SPACE, dtype=bool)
                mask[ACTION_AGARI] = True
                return (obs_snap, ACTION_AGARI, mask)

            # Fallback
            obs_snap = self._encode_obs_snapshot()
            mask = self._make_mask_for_agari()
            self._pending_decision = None
            self._pending_call_decision = None
            return (obs_snap, ACTION_AGARI, mask)

        else:
            # Another player won → emit pass sample if we had a callable opportunity
            return self._emit_pass_if_pending()

    def _on_ryukyoku(self, ev: dict) -> Optional[tuple[np.ndarray, int, np.ndarray]]:
        return self._emit_pass_if_pending()

    # ------------------------------------------------------------------
    # Helper: update shanten
    # ------------------------------------------------------------------

    # Yaochuu tile indices: 1m/9m/1p/9p/1s/9s + honours (27-33)
    _YAOCHUU_SET = frozenset([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33])

    def _update_shanten(self) -> None:
        from . import obs_encoder as _enc
        if not _enc.USE_SHANTEN_FEATURES:
            self.shanten = 0
            return
        """
        Compute weighted shanten mirroring Bot.vala analyze_hand logic.

        weighted_shanten = raw_shanten + penalty, where penalty is:
          +2  if raw_shanten >= 3  and hand violates win prerequisites
          +2  if raw_shanten <  3  and hand violates win prerequisites  (Vala uses +2 here too)
          +1  if raw_shanten >= 3  and hand violates win prerequisites  (Vala: shanten_num + 1)
           0  otherwise (hand looks valid)

        Win prerequisites (checked on closed hand + open melds):
          1. At least one triplet (pon/kan meld, or 3+ of same tile in hand)
          2. At least one terminal or dragon tile
          3. Not honitsu/chinitsu (more than one suit category present,
             ignoring honour tiles which don't count as a suit)
          4. Not all-triplets (triplet_count must be < 4; pure toitoi is invalid)

        Vala formula (lines 566-581):
          if (no_terminals_dragons) OR (single_suit) OR (no_triplets) OR (all_triplets):
              weighted = raw + 1  if raw >= 3
              weighted = raw + 2  if raw < 3
          else:
              weighted = raw
        """
        from .shanten import calc_shanten_cached

        tile_list = self.tehai.tolist()
        open_count = sum(len(m) // 3 for m in self.fuuro[self.player_id])
        raw = calc_shanten_cached(tile_list, open_count)

        # --- terminal/dragon count across hand + open melds ---
        terminal_dragon = 0
        for idx in self._YAOCHUU_SET:
            terminal_dragon += self.tehai[idx]
        for meld in self.fuuro[self.player_id]:
            for t in meld:
                if t in self._YAOCHUU_SET:
                    terminal_dragon += 1
        for t in self.ankan[self.player_id]:
            if t in self._YAOCHUU_SET:
                terminal_dragon += 4

        # --- triplet count across hand + open melds ---
        triplet_count = 0
        for i in range(TILE_COUNT):
            if self.tehai[i] >= 3:
                triplet_count += 1
        for meld in self.fuuro[self.player_id]:
            if len(meld) >= 3 and len(set(meld[:3])) == 1:
                triplet_count += 1
        triplet_count += len(self.ankan[self.player_id])

        # --- suit categories (excluding honours) ---
        suit_cats: set[int] = set()
        for i in range(27):
            if self.tehai[i] > 0:
                suit_cats.add(i // 9)
        for meld in self.fuuro[self.player_id]:
            for t in meld:
                if t < 27:
                    suit_cats.add(t // 9)

        # --- sequence detection: open melds + hand partial runs ---
        has_sequence = False
        # Check open melds for any chii (3 distinct tiles in same suit, consecutive)
        for meld in self.fuuro[self.player_id]:
            if len(meld) >= 3:
                a, b, c = sorted(meld[:3])
                if a < 27 and (a // 9 == b // 9 == c // 9) and b == a + 1 and c == a + 2:
                    has_sequence = True
                    break
        # Check hand tiles for any partial sequence (two suit tiles within distance 2)
        if not has_sequence:
            suit_tiles = [i for i in range(27) if self.tehai[i] > 0]
            for i in range(len(suit_tiles)):
                for j in range(i + 1, len(suit_tiles)):
                    a, b = suit_tiles[i], suit_tiles[j]
                    if a // 9 == b // 9 and b - a <= 2:
                        has_sequence = True
                        break
                if has_sequence:
                    break

        no_terminal_dragon = (terminal_dragon == 0)
        single_suit = (len(suit_cats) <= 1)
        no_triplet = (triplet_count <= 0)
        no_sequence = not has_sequence

        if no_terminal_dragon or single_suit or no_triplet or no_sequence:
            self.shanten = raw + 1 if raw >= 3 else raw + 2
        else:
            self.shanten = raw

    def _calc_post_call_shanten(self, called_tile: int, variant: str) -> int:
        from . import obs_encoder as _enc
        if not _enc.USE_SHANTEN_FEATURES:
            return 0
        """
        Compute weighted_shanten on a hypothetical hand after making a call.
        variant: 'chi_low'|'chi_mid'|'chi_high'|'pon'|'kan'
        Returns the weighted shanten value without modifying state.
        """
        from .shanten import calc_shanten_cached

        tehai = self.tehai.copy()
        fuuro = [list(m) for m in self.fuuro[self.player_id]]

        # Apply the hypothetical call: add called tile, remove consumed tiles, add meld
        if variant in ('chi_low', 'chi_mid', 'chi_high'):
            if variant == 'chi_low':
                consumed = [called_tile + 1, called_tile + 2]
            elif variant == 'chi_mid':
                consumed = [called_tile - 1, called_tile + 1]
            else:
                consumed = [called_tile - 2, called_tile - 1]
            for t in consumed:
                if tehai[t] > 0:
                    tehai[t] -= 1
            meld = sorted([called_tile] + consumed)
            fuuro.append(meld)
        elif variant == 'pon':
            tehai[called_tile] = max(0, tehai[called_tile] - 2)
            fuuro.append([called_tile, called_tile, called_tile])
        elif variant == 'kan':
            tehai[called_tile] = max(0, tehai[called_tile] - 3)
            fuuro.append([called_tile, called_tile, called_tile, called_tile])

        open_count = sum(len(m) // 3 for m in fuuro)
        raw = calc_shanten_cached(tehai.tolist(), open_count)

        # terminal/dragon
        terminal_dragon = 0
        for idx in self._YAOCHUU_SET:
            terminal_dragon += tehai[idx]
        for meld in fuuro:
            for t in meld:
                if t in self._YAOCHUU_SET:
                    terminal_dragon += 1

        # triplet count
        triplet_count = 0
        for i in range(TILE_COUNT):
            if tehai[i] >= 3:
                triplet_count += 1
        for meld in fuuro:
            if len(meld) >= 3 and len(set(meld[:3])) == 1:
                triplet_count += 1

        # suit categories
        suit_cats: set[int] = set()
        for i in range(27):
            if tehai[i] > 0:
                suit_cats.add(i // 9)
        for meld in fuuro:
            for t in meld:
                if t < 27:
                    suit_cats.add(t // 9)

        # sequence detection
        has_sequence = False
        for meld in fuuro:
            if len(meld) >= 3:
                a, b, c = sorted(meld[:3])
                if a < 27 and (a // 9 == b // 9 == c // 9) and b == a + 1 and c == a + 2:
                    has_sequence = True
                    break
        if not has_sequence:
            suit_tiles = [i for i in range(27) if tehai[i] > 0]
            for i in range(len(suit_tiles)):
                for j in range(i + 1, len(suit_tiles)):
                    a, b = suit_tiles[i], suit_tiles[j]
                    if a // 9 == b // 9 and b - a <= 2:
                        has_sequence = True
                        break
                if has_sequence:
                    break

        no_terminal_dragon = (terminal_dragon == 0)
        single_suit = (len(suit_cats) <= 1)
        no_triplet = (triplet_count <= 0)
        no_sequence = not has_sequence

        if no_terminal_dragon or single_suit or no_triplet or no_sequence:
            return raw + 1 if raw >= 3 else raw + 2
        return raw

    # ------------------------------------------------------------------
    # Helper: snapshot for training
    # ------------------------------------------------------------------

    def _snapshot_decision(self) -> tuple[np.ndarray, np.ndarray]:
        """Capture (obs, mask) at decision time."""
        return (self._encode_obs_snapshot(), self._make_mask())

    def _make_mask(self) -> np.ndarray:
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        if self._can_discard:
            for i in range(TILE_COUNT):
                if self.tehai[i] > 0:
                    mask[i] = True
        if self._can_chi_low:
            mask[ACTION_CHI_LOW] = True
        if self._can_chi_mid:
            mask[ACTION_CHI_MID] = True
        if self._can_chi_high:
            mask[ACTION_CHI_HIGH] = True
        if self._can_pon:
            mask[ACTION_PON] = True
        if self._can_kan:
            mask[ACTION_KAN] = True
        if self._can_agari:
            mask[ACTION_AGARI] = True
        if self._can_ryukyoku:
            mask[ACTION_RYUKYOKU] = True
        return mask

    def _make_mask_for_agari(self) -> np.ndarray:
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        mask[ACTION_AGARI] = True
        return mask

    def _encode_obs_snapshot(self) -> np.ndarray:
        """Return obs array snapshot; import here to avoid circular dependency."""
        from .obs_encoder import encode_obs
        obs = encode_obs(self)
        if self._obs_hook is not None:
            self._obs_hook(self, obs)
        return obs
