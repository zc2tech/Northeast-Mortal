#!/usr/bin/env python3
"""
Rehearsal test: verify that the MortalBot snapshot path produces identical obs
to the mjai log replay path (training path).

Uses _obs_hook to capture the exact NortheastPlayerState at the moment the
training obs is encoded, then reconstructs the snapshot from that state and
verifies the inference path produces the same obs.
"""

import gzip
import json
import sys
import os
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_MORTAL = os.path.join(_ROOT, "mortal")
if _MORTAL not in sys.path:
    sys.path.insert(0, _MORTAL)

from libne.player_state import NortheastPlayerState, KawaItem
from libne.tile import mjai_to_idx
from libne.consts import WIND_INDEX, TILE_COUNT


# ── apply_snapshot (mirrors inference_server.py exactly) ─────────────────────

def apply_snapshot(state: NortheastPlayerState, ev: dict) -> None:
    state._reset()
    pid = ev.get("player_id", state.player_id)
    state.player_id = pid
    state.dealer = ev.get("dealer", 0)
    state.bakaze = WIND_INDEX.get(ev.get("round_wind", "E"), WIND_INDEX["E"])
    state.kyoku  = ev.get("kyoku", 1) - 1
    seat_wind_names = ["E", "S", "W", "N"]
    seat_offset = (pid - state.dealer) % 4
    state.jikaze = WIND_INDEX[seat_wind_names[seat_offset]]
    raw_scores = ev.get("scores", [0, 0, 0, 0])
    for i in range(4):
        state.scores[i] = raw_scores[(pid + i) % 4]
    for tile_str in ev.get("hands", [[], [], [], []])[pid]:
        idx = mjai_to_idx(tile_str)
        if idx >= 0:
            state.tehai[idx] += 1
    ponds = ev.get("ponds", [[], [], [], []])
    for p in range(4):
        for tile_str in ponds[p]:
            idx = mjai_to_idx(tile_str)
            state.kawa[p].append(KawaItem(idx))
    calls_data = ev.get("calls", [[], [], [], []])
    for p in range(4):
        for call in calls_data[p]:
            tiles_idx = [mjai_to_idx(s) for s in call.get("tiles", [])]
            ctype = call.get("call_type", "")
            if ctype == "closed_kan":
                state.ankan[p].append(tiles_idx[0] if tiles_idx else -1)
            else:
                state.fuuro[p].append(tiles_idx)
    total_discards = sum(len(state.kawa[p]) for p in range(4))
    # Use explicit tiles_left if sent, otherwise approximate from discards
    if "tiles_left" in ev:
        state.tiles_left = ev["tiles_left"]
    else:
        state.tiles_left = max(0, 69 - total_discards)
    state.at_turn    = total_discards
    dwm_str = ev.get("dead_wall_marker", "")
    if dwm_str:
        idx = mjai_to_idx(dwm_str)
        if idx >= 0:
            state.dead_wall_markers = [idx]
    last_discard_str = ev.get("last_discard")
    if last_discard_str:
        idx = mjai_to_idx(last_discard_str)
        if idx >= 0:
            state.last_kawa_tile = idx
            state.last_kawa_player = ev.get("last_discard_player", state.last_kawa_player)
    state._can_discard = True
    state._update_shanten()


def _has_chi_group(state, variant):
    tile = state.last_kawa_tile
    if tile < 0 or tile > 26:
        return False
    suit_base = (tile // 9) * 9
    suit_end  = suit_base + 8
    if variant == 0:   a, b = tile+1, tile+2
    elif variant == 1: a, b = tile-1, tile+1
    else:              a, b = tile-2, tile-1
    if a < suit_base or b < suit_base or a > suit_end or b > suit_end:
        return False
    return state.tehai[a] >= 1 and state.tehai[b] >= 1


def set_action_flags(inf_state, src_state):
    """Copy exact _can_* flags and post_call_shanten from the hook-captured state."""
    inf_state._can_discard   = src_state._can_discard
    inf_state._can_chi_low   = src_state._can_chi_low
    inf_state._can_chi_mid   = src_state._can_chi_mid
    inf_state._can_chi_high  = src_state._can_chi_high
    inf_state._can_pon       = src_state._can_pon
    inf_state._can_kan       = src_state._can_kan
    inf_state._can_agari     = src_state._can_agari
    inf_state._can_ryukyoku  = src_state._can_ryukyoku
    inf_state.post_call_shanten = list(src_state.post_call_shanten)


# ── Snapshot builder from hook-captured state ─────────────────────────────────

def tile_to_mjai(idx):
    if 0 <= idx <= 8:   return f"{idx+1}m"
    if 9 <= idx <= 17:  return f"{idx-9+1}p"
    if 18 <= idx <= 26: return f"{idx-18+1}s"
    return {27:"E",28:"S",29:"W",30:"N",31:"P",32:"F",33:"C"}.get(idx,"?")


def wind_str(bakaze_idx):
    return {27:"E",28:"S",29:"W",30:"N"}.get(bakaze_idx,"E")


def build_snapshot(s: NortheastPlayerState, last_discard_tile: int, last_discard_player: int) -> dict:
    """Build the snapshot dict that MortalBot would send, from the hook-captured state."""
    pid = s.player_id
    abs_scores = [s.scores[(i - pid) % 4] for i in range(4)]

    hands = [[], [], [], []]
    for tile_idx in range(TILE_COUNT):
        for _ in range(int(s.tehai[tile_idx])):
            hands[pid].append(tile_to_mjai(tile_idx))

    ponds = []
    for p in range(4):
        ponds.append([tile_to_mjai(item.tile) for item in s.kawa[p]])

    calls = [[], [], [], []]
    for p in range(4):
        for meld in s.fuuro[p]:
            if len(meld) == 4:
                ct = "open_kan"
            elif len(meld) == 3 and len(set(meld)) == 1:
                ct = "pon"
            elif len(meld) == 3:
                ct = "chi"
            else:
                ct = "unknown"
            calls[p].append({"call_type": ct, "tiles": [tile_to_mjai(t) for t in meld]})
        for t in s.ankan[p]:
            calls[p].append({"call_type": "closed_kan", "tiles": [tile_to_mjai(t)]*4})

    snap = {
        "type": "snapshot",
        "player_id": pid,
        "dealer": s.dealer,
        "round_wind": wind_str(s.bakaze),
        "kyoku": s.kyoku + 1,
        "scores": abs_scores,
        "hands": hands,
        "ponds": ponds,
        "calls": calls,
        # Send tiles_left explicitly so server doesn't recompute from pond sizes alone
        # (training decrements on tsumo draws, which pond sizes don't capture)
        "tiles_left": s.tiles_left,
    }
    if s.dead_wall_markers:
        snap["dead_wall_marker"] = tile_to_mjai(s.dead_wall_markers[0])
    # Use the state's last_kawa_tile directly (already set correctly for both
    # turn and call decisions at the moment the obs was captured)
    if s.last_kawa_tile >= 0:
        snap["last_discard"] = tile_to_mjai(s.last_kawa_tile)
        snap["last_discard_player"] = s.last_kawa_player
    return snap


# ── Rehearsal engine ──────────────────────────────────────────────────────────

def run_rehearsal(log_path: str) -> None:
    with gzip.open(log_path, "rt", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    kyoku_events = []
    current_kyoku: list = []
    for ev in records:
        t = ev.get("type", "")
        if t == "start_game":
            continue
        if t == "start_kyoku":
            if current_kyoku:
                kyoku_events.append(current_kyoku)
            current_kyoku = [ev]
        else:
            current_kyoku.append(ev)
    if current_kyoku:
        kyoku_events.append(current_kyoku)

    total_checked = 0
    total_mismatch = 0
    mismatch_details = []

    for kyoku_idx, events in enumerate(kyoku_events):
        for pid in range(4):
            train_state = NortheastPlayerState(pid)

            # MortalBot's tracked last discard (updated on every dahai)
            mb_last_discard_tile   = -1
            mb_last_discard_player = -1

            # Hook: fires inside _encode_obs_snapshot exactly when training obs is captured.
            # We store both the obs array and state so we can match by obs identity.
            capture_queue: list = []

            def obs_hook(s, obs, _q=capture_queue):
                _q.append({
                    "obs": obs,   # keep reference for matching
                    "state_tehai":   s.tehai.copy(),
                    "state_kawa":    [[KawaItem(x.tile, x.from_kan) for x in row] for row in s.kawa],
                    "state_fuuro":   [[list(m) for m in p] for p in s.fuuro],
                    "state_ankan":   [list(a) for a in s.ankan],
                    "state_scores":  list(s.scores),
                    "state_bakaze":  s.bakaze,
                    "state_jikaze":  s.jikaze,
                    "state_kyoku":   s.kyoku,
                    "state_dealer":  s.dealer,
                    "state_tiles_left": s.tiles_left,
                    "state_dwm":     list(s.dead_wall_markers),
                    "state_last_kawa_tile":   s.last_kawa_tile,
                    "state_last_kawa_player": s.last_kawa_player,
                    "can_discard":   s._can_discard,
                    "can_chi_low":   s._can_chi_low,
                    "can_chi_mid":   s._can_chi_mid,
                    "can_chi_high":  s._can_chi_high,
                    "can_pon":       s._can_pon,
                    "can_kan":       s._can_kan,
                    "can_agari":     s._can_agari,
                    "can_ryukyoku":  s._can_ryukyoku,
                    "post_call_shanten": list(s.post_call_shanten),
                    "pid":           s.player_id,
                })

            train_state._obs_hook = obs_hook

            for ev in events:
                etype = ev.get("type", "")

                result = train_state.update(ev)

                # Clear stale captures and last_discard on kyoku boundary
                if etype == "start_kyoku":
                    capture_queue.clear()
                    mb_last_discard_tile   = -1
                    mb_last_discard_player = -1

                # Update MortalBot's last_discard after every dahai
                if etype == "dahai":
                    mb_last_discard_tile   = mjai_to_idx(ev.get("pai", ""))
                    mb_last_discard_player = ev.get("actor", -1)

                if result is None:
                    continue

                obs_train, action_label, mask_train = result

                # Find the matching capture by obs array identity
                # (the training obs is one of the encoded obs in capture_queue)
                c = None
                for i, cap in enumerate(capture_queue):
                    if cap["obs"] is obs_train:
                        c = capture_queue.pop(i)
                        break
                if c is None:
                    continue

                # Reconstruct the snapshot from hook-captured state
                # Build a temporary state object to pass to build_snapshot
                tmp = NortheastPlayerState(c["pid"])
                tmp.tehai         = c["state_tehai"]
                tmp.kawa          = c["state_kawa"]
                tmp.fuuro         = c["state_fuuro"]
                tmp.ankan         = c["state_ankan"]
                tmp.scores        = c["state_scores"]
                tmp.bakaze        = c["state_bakaze"]
                tmp.jikaze        = c["state_jikaze"]
                tmp.kyoku         = c["state_kyoku"]
                tmp.dealer        = c["state_dealer"]
                tmp.tiles_left    = c["state_tiles_left"]
                tmp.dead_wall_markers    = c["state_dwm"]
                tmp.last_kawa_tile       = c["state_last_kawa_tile"]
                tmp.last_kawa_player     = c["state_last_kawa_player"]

                snap = build_snapshot(tmp, -1, -1)

                # Inference path
                inf_state = NortheastPlayerState(c["pid"])
                apply_snapshot(inf_state, snap)

                # Copy exact action flags from hook (what server sets before encoding)
                inf_state._can_discard   = c["can_discard"]
                inf_state._can_chi_low   = c["can_chi_low"]
                inf_state._can_chi_mid   = c["can_chi_mid"]
                inf_state._can_chi_high  = c["can_chi_high"]
                inf_state._can_pon       = c["can_pon"]
                inf_state._can_kan       = c["can_kan"]
                inf_state._can_agari     = c["can_agari"]
                inf_state._can_ryukyoku  = c["can_ryukyoku"]
                inf_state.post_call_shanten = c["post_call_shanten"]

                obs_inf = inf_state._encode_obs_snapshot()

                total_checked += 1
                if not np.allclose(obs_train, obs_inf, atol=1e-5):
                    total_mismatch += 1
                    diff_mask = ~np.isclose(obs_train, obs_inf, atol=1e-5)
                    diff_rows = sorted(set(np.where(diff_mask)[0].tolist()))
                    if len(mismatch_details) < 20:
                        examples = []
                        for r in diff_rows[:5]:
                            cols = np.where(~np.isclose(obs_train[r], obs_inf[r], atol=1e-5))[0]
                            examples.append(
                                f"row{r} col{cols[0]}: train={obs_train[r,cols[0]]:.4f} inf={obs_inf[r,cols[0]]:.4f}"
                            )
                        mismatch_details.append({
                            "kyoku": kyoku_idx, "pid": pid,
                            "action": action_label, "rows": diff_rows, "ex": examples,
                        })

    print(f"\n{'='*60}")
    print(f"Rehearsal: {total_checked} decision points, {total_mismatch} mismatches")
    if mismatch_details:
        print(f"\nFirst {len(mismatch_details)} mismatches:")
        for m in mismatch_details:
            print(f"  kyoku={m['kyoku']} pid={m['pid']} action={m['action']} rows={m['rows']}")
            for ex in m["ex"]:
                print(f"    {ex}")
    else:
        print("All obs arrays match exactly. Snapshot faithfully recreates training obs.")
    print('='*60)


if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/I572958/.config/Northeast-Mahjong/logs/mjai/2026-08-25_13-06-52_aug1.json.gz"
    print(f"Rehearsing: {log}")
    run_rehearsal(log)
