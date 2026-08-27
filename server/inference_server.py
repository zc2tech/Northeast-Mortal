#!/usr/bin/env python3
"""
Northeast Mahjong Mortal inference server.

Listens on TCP (default 127.0.0.1:11617).
Each connection handles one bot seat. The client (MortalBot in Vala) sends
mjai JSON lines and the server responds with the chosen action as JSON.

Protocol (per connection):
  Client → Server:  JSON line for each game event
  Server → Client:  JSON response only when the bot must make a decision
      {"type":"action","action":<int>}   action index (0-45, see libne/consts.py)

The client sends a {"type":"hello","player_id":<0-3>} line first to register
the seat. After that, game events flow in mjai format. The server maintains one
NortheastPlayerState per connection and runs Brain+DQN inference on obs/mask.

Run:
    cd /path/to/Northeast-Mortal
    python server/inference_server.py [--host 127.0.0.1] [--port 11617] \
        [--model /path/to/mortal.pth] [--device cpu]
"""

import argparse
import json
import os
import socket
import sys
import threading

import numpy as np
import torch

# Ensure Northeast-Mortal root is importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# mortal/ must also be on path for model.py imports
_MORTAL = os.path.join(_ROOT, "mortal")
if _MORTAL not in sys.path:
    sys.path.insert(0, _MORTAL)

from libne.player_state import NortheastPlayerState
from libne.consts import ACTION_SPACE, OBS_ROWS, OBS_COLS

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mortal.pth")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11617


def load_model(path: str, device: torch.device):
    state = torch.load(path, weights_only=True, map_location=device)
    cfg = state["config"]
    version     = cfg["control"].get("version", 4)
    num_blocks  = cfg["resnet"]["num_blocks"]
    conv_ch     = cfg["resnet"]["conv_channels"]

    # Import Brain/DQN after mortal/ is on path
    from model import Brain, DQN
    brain = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_ch).to(device).eval()
    dqn   = DQN(version=version).to(device).eval()
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    print(f"[server] loaded mortal.pth  version={version} blocks={num_blocks} ch={conv_ch}", flush=True)
    return brain, dqn, version


def choose_action(brain, dqn, version: int, obs: np.ndarray, mask: np.ndarray, device: torch.device) -> int:
    """Run Brain → DQN → argmax, return action index."""
    obs_t  = torch.as_tensor(obs[np.newaxis],  dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask[np.newaxis], dtype=torch.bool,    device=device)

    with torch.inference_mode():
        match version:
            case 1:
                mu, _ = brain(obs_t, None)
                phi = mu
            case 2 | 3 | 4:
                phi = brain(obs_t)
            case _:
                phi = brain(obs_t)

        q_out = dqn(phi, mask_t)          # (1, 46)
        # mask out illegal actions
        q_out = q_out.masked_fill(~mask_t, -torch.inf)
        action = int(q_out.argmax(dim=-1).item())

    return action


def _can_ron(state: "NortheastPlayerState") -> bool:
    """Allow ron on call decisions — game engine validates the actual legality."""
    return state.last_kawa_tile >= 0


def round_state_can_tsumo(state: "NortheastPlayerState") -> bool:
    """Allow tsumo on turn decisions — game engine validates the actual legality."""
    return True


def _has_chi_group(state: "NortheastPlayerState", variant: int) -> bool:
    """
    Check if a chi of the given variant (0=low,1=mid,2=high) is possible.
    variant 0: called tile is lowest (hand has mid+high)
    variant 1: called tile is middle (hand has low+high)
    variant 2: called tile is highest (hand has low+mid)
    Only valid for suit tiles (0-26).
    """
    tile = state.last_kawa_tile
    if tile < 0 or tile > 26:
        return False
    suit_base = (tile // 9) * 9
    suit_end  = suit_base + 8
    num = tile - suit_base  # 0-8

    if variant == 0:   # called=low; need mid(+1) and high(+2)
        a, b = tile + 1, tile + 2
    elif variant == 1: # called=mid; need low(-1) and high(+1)
        a, b = tile - 1, tile + 1
    else:              # called=high; need low(-2) and mid(-1)
        a, b = tile - 2, tile - 1

    if a < suit_base or b < suit_base or a > suit_end or b > suit_end:
        return False
    return state.tehai[a] >= 1 and state.tehai[b] >= 1


def apply_snapshot(state: "NortheastPlayerState", ev: dict) -> None:
    """
    Populate NortheastPlayerState directly from a full game-state snapshot
    sent by MortalBot, bypassing the mjai event stream.
    """
    from libne.tile import mjai_to_idx
    from libne.consts import WIND_INDEX, TILE_COUNT
    import numpy as np

    state._reset()

    pid = ev.get("player_id", state.player_id)
    state.player_id = pid

    # Round info
    state.dealer = ev.get("dealer", 0)
    state.bakaze = WIND_INDEX.get(ev.get("round_wind", "E"), WIND_INDEX["E"])
    state.kyoku  = ev.get("kyoku", 1) - 1   # store 0-indexed

    # Seat wind
    seat_wind_names = ["E", "S", "W", "N"]
    seat_offset = (pid - state.dealer) % 4
    from libne.consts import WIND_INDEX as WI
    state.jikaze = WI[seat_wind_names[seat_offset]]

    # Scores (rotate so self = index 0)
    raw_scores = ev.get("scores", [0, 0, 0, 0])
    for i in range(4):
        state.scores[i] = raw_scores[(pid + i) % 4]

    # Own hand
    for tile_str in ev.get("hands", [[], [], [], []])[pid]:
        idx = mjai_to_idx(tile_str)
        if idx >= 0:
            state.tehai[idx] += 1

    # Ponds (discard rivers) — do NOT derive last_kawa_tile here;
    # the explicit last_discard field below is authoritative.
    from libne.player_state import KawaItem
    ponds = ev.get("ponds", [[], [], [], []])
    for p in range(4):
        for tile_str in ponds[p]:
            idx = mjai_to_idx(tile_str)
            state.kawa[p].append(KawaItem(idx))

    # Open melds / calls
    calls_data = ev.get("calls", [[], [], [], []])
    for p in range(4):
        for call in calls_data[p]:
            tiles_idx = [mjai_to_idx(s) for s in call.get("tiles", [])]
            ctype = call.get("call_type", "")
            if ctype == "closed_kan":
                state.ankan[p].append(tiles_idx[0] if tiles_idx else -1)
            else:
                state.fuuro[p].append(tiles_idx)

    # Tiles left approximation: use explicit value if sent, else derive from discards
    total_discards = sum(len(state.kawa[p]) for p in range(4))
    if "tiles_left" in ev:
        state.tiles_left = ev["tiles_left"]
    else:
        state.tiles_left = max(0, 69 - total_discards)
    state.at_turn    = total_discards

    # Dead wall marker
    dwm_str = ev.get("dead_wall_marker", "")
    if dwm_str:
        from libne.tile import mjai_to_idx as _mjai_to_idx
        idx = _mjai_to_idx(dwm_str)
        if idx >= 0:
            state.dead_wall_markers = [idx]

    # Override last_kawa_tile from explicit last_discard field if present
    # (the discarded tile may not be in the pond yet during call_decision)
    last_discard_str = ev.get("last_discard")
    if last_discard_str:
        idx = mjai_to_idx(last_discard_str)
        if idx >= 0:
            state.last_kawa_tile = idx
            state.last_kawa_player = ev.get("last_discard_player", state.last_kawa_player)

    # Set discard flag; will be overridden by decision type
    state._can_discard = True

    # Compute weighted shanten now that hand + melds are fully populated
    state._update_shanten()


def handle_connection(conn: socket.socket, addr, brain, dqn, version: int, device: torch.device):
    print(f"[server] connection from {addr}", flush=True)
    player_id = 0
    state: NortheastPlayerState | None = None
    buf = b""

    try:
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[server] bad JSON from {addr}: {e}", flush=True)
                    continue

                etype = event.get("type", "")

                # Registration handshake
                if etype == "hello":
                    player_id = int(event.get("player_id", 0))
                    state = NortheastPlayerState(player_id)
                    resp = json.dumps({"type": "hello_ack", "player_id": player_id})
                    conn.sendall((resp + "\n").encode())
                    print(f"[server] seat {player_id} registered from {addr}", flush=True)
                    continue

                # Reset on new game
                if etype == "start_game":
                    if state is None:
                        state = NortheastPlayerState(player_id)
                    else:
                        state._reset()
                    continue

                if state is None:
                    continue

                # Full state snapshot sent by MortalBot before each decision
                if etype == "snapshot":
                    apply_snapshot(state, event)
                    continue

                if etype in ("turn_decision", "call_decision"):
                    # Set legal action flags based on decision type
                    if etype == "turn_decision":
                        state._can_discard   = True
                        state._can_chi_low   = False
                        state._can_chi_mid   = False
                        state._can_chi_high  = False
                        state._can_pon       = False
                        state._can_kan       = False
                        state._can_agari     = round_state_can_tsumo(state)
                        state._can_ryukyoku  = False
                        state.post_call_shanten = [-1, -1, -1, -1, -1]
                    else:
                        # call_decision: can potentially call on last kawa tile
                        state._can_discard   = False
                        state._can_chi_low   = _has_chi_group(state, 0)
                        state._can_chi_mid   = _has_chi_group(state, 1)
                        state._can_chi_high  = _has_chi_group(state, 2)
                        state._can_pon       = state.tehai[state.last_kawa_tile] >= 2 if state.last_kawa_tile >= 0 else False
                        state._can_kan       = state.tehai[state.last_kawa_tile] >= 3 if state.last_kawa_tile >= 0 else False
                        state._can_agari     = _can_ron(state)
                        state._can_ryukyoku  = False
                        tile = state.last_kawa_tile
                        hand_near = [state.tehai[t] for t in range(max(0,tile-2), min(34,tile+3))] if tile >= 0 else []
                        print(f"[server] call_decision: last_kawa_tile={tile} "
                              f"pon={state._can_pon} chi={state._can_chi_low}/{state._can_chi_mid}/{state._can_chi_high} "
                              f"ron={state._can_agari} hand_near(t-2..t+2)={hand_near}", flush=True)
                        # Compute post-call shanten so obs rows 206-210 match training
                        from libne import obs_encoder as _enc
                        if _enc.USE_SHANTEN_FEATURES and tile >= 0:
                            state.post_call_shanten = [
                                state._calc_post_call_shanten(tile, 'chi_low')  if state._can_chi_low  else -1,
                                state._calc_post_call_shanten(tile, 'chi_mid')  if state._can_chi_mid  else -1,
                                state._calc_post_call_shanten(tile, 'chi_high') if state._can_chi_high else -1,
                                state._calc_post_call_shanten(tile, 'pon')      if state._can_pon      else -1,
                                state._calc_post_call_shanten(tile, 'kan')      if state._can_kan      else -1,
                            ]
                        else:
                            state.post_call_shanten = [-1, -1, -1, -1, -1]

                    obs  = state._encode_obs_snapshot()
                    mask = state._make_mask()

                    # Always allow pass on call decision; fallback if nothing legal
                    if etype == "call_decision":
                        mask[45] = True  # ACTION_PASS always available
                    if not mask.any():
                        mask[45] = True

                    action = choose_action(brain, dqn, version, obs, mask, device)
                    print(f"[server] {etype}: mask_sum={mask.sum()} action={action}", flush=True)
                    resp = json.dumps({"type": "action", "action": action})
                    conn.sendall((resp + "\n").encode())
                    continue

                # Forward other mjai events to player state (for any incremental tracking)
                state.update(event)

    except Exception as e:
        import traceback
        print(f"[server] error handling {addr}: {e}\n{traceback.format_exc()}", flush=True)
    finally:
        conn.close()
        print(f"[server] connection closed {addr}", flush=True)


def serve(host: str, port: int, brain, dqn, version: int, device: torch.device):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    print(f"[server] listening on {host}:{port}  (press Ctrl+C to stop)", flush=True)

    try:
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(
                target=handle_connection,
                args=(conn, addr, brain, dqn, version, device),
                daemon=True,
            )
            t.start()
    except KeyboardInterrupt:
        print("\n[server] shutting down...", flush=True)
    finally:
        srv.close()
        print("[server] stopped", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Northeast Mahjong Mortal inference server")
    parser.add_argument("--host",   default=DEFAULT_HOST)
    parser.add_argument("--port",   type=int, default=DEFAULT_PORT)
    parser.add_argument("--model",  default=None,
                        help="Path to mortal.pth (default: bin/Data/Shanten/mortal.pth)")
    parser.add_argument("--device", default="cpu",
                        help="torch device: cpu / cuda / mps")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        model_path = DEFAULT_MODEL

    device = torch.device(args.device)
    brain, dqn, version = load_model(model_path, device)

    serve(args.host, args.port, brain, dqn, version, device)


if __name__ == "__main__":
    main()
