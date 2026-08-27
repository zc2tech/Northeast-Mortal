"""
NortheastGameplayLoader — replaces libriichi's GameplayLoader.
Reads .json.gz mjai log files and emits NortheastGameplay objects
with the same take_*() interface that mortal/train.py expects.
"""

from __future__ import annotations

import gzip
import json
import numpy as np
from pathlib import Path
from typing import Optional

from .player_state import NortheastPlayerState
from .consts import ACTION_SPACE


class NortheastGrp:
    """
    Game Result Predictor data.
    Stores per-kyoku point deltas and final rank for each player seat.
    Mortal's GRP takes a (GRP_SIZE, 4) feature array.
    We store the deltas as the feature directly.
    """

    def __init__(self) -> None:
        self._feature: list[np.ndarray] = []   # list of (4,) float32 arrays
        self._rank_by_player: list[int] = []    # final ranks 0-3 for each player seat
        self._final_scores: list[int] = []

    def add_kyoku(self, deltas: list[int]) -> None:
        self._feature.append(np.array(deltas, dtype=np.float32) / 100.0)

    def set_final(self, scores: list[int]) -> None:
        self._final_scores = scores[:]
        sorted_seats = sorted(range(4), key=lambda i: -scores[i])
        self._rank_by_player = [0] * 4
        for rank, seat in enumerate(sorted_seats):
            self._rank_by_player[seat] = rank

    def take_feature(self) -> np.ndarray:
        """Returns (N_kyoku, 4) float32 array of per-kyoku normalised deltas."""
        if not self._feature:
            return np.zeros((1, 4), dtype=np.float32)
        return np.stack(self._feature)

    def take_rank_by_player(self) -> list[int]:
        return self._rank_by_player or [0, 1, 2, 3]

    def take_final_scores(self) -> list[int]:
        return self._final_scores or [0, 0, 0, 0]


class NortheastGameplay:
    """
    Mirrors libriichi's Gameplay class.
    Stores all decision records for one player across one game.
    """

    def __init__(self, player_id: int) -> None:
        self.player_id = player_id
        self._obs: list[np.ndarray] = []
        self._actions: list[int] = []
        self._masks: list[np.ndarray] = []
        self._at_kyoku: list[int] = []
        self._dones: list[bool] = []
        self._apply_gamma: list[bool] = []
        self._at_turns: list[int] = []
        self._shantens: list[int] = []
        self._grp = NortheastGrp()
        self._current_kyoku = 0

    def add_record(self, obs: np.ndarray, action: int, mask: np.ndarray,
                   at_turn: int, shanten: int, done: bool) -> None:
        # Always ensure the action taken is valid in the mask.
        # Call-events (chi/pon/agari) may snapshot before their flags are set.
        mask = mask.copy()
        if 0 <= action < len(mask):
            mask[action] = True
        self._obs.append(obs)
        self._actions.append(action)
        self._masks.append(mask)
        self._at_kyoku.append(self._current_kyoku)
        self._dones.append(done)
        self._apply_gamma.append(action <= 37)   # discard/riichi → apply gamma
        self._at_turns.append(at_turn)
        self._shantens.append(shanten)

    # take_* methods mirror libriichi's Gameplay Python API
    def take_obs(self) -> list[np.ndarray]:
        return self._obs

    def take_actions(self) -> list[int]:
        return self._actions

    def take_masks(self) -> list[np.ndarray]:
        return self._masks

    def take_at_kyoku(self) -> list[int]:
        return self._at_kyoku

    def take_dones(self) -> list[bool]:
        return self._dones

    def take_apply_gamma(self) -> list[bool]:
        return self._apply_gamma

    def take_at_turns(self) -> list[int]:
        return self._at_turns

    def take_shantens(self) -> list[int]:
        return self._shantens

    def take_grp(self) -> NortheastGrp:
        return self._grp

    def take_player_id(self) -> int:
        return self.player_id


class NortheastGameplayLoader:
    """
    Replaces libriichi's GameplayLoader.
    Parses .json.gz mjai log files and returns a list of NortheastGameplay objects
    (one per player seat per game file).
    """

    def __init__(self) -> None:
        pass

    def load_gz_log_files(self, file_list: list[str]) -> list[NortheastGameplay]:
        results: list[NortheastGameplay] = []
        for path in file_list:
            try:
                gameplays = self._load_one(path)
                results.extend(gameplays)
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to load {path}: {e}")
        return results

    def _load_one(self, path: str) -> list[NortheastGameplay]:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]

        # Create one state + gameplay per player seat
        states = [NortheastPlayerState(i) for i in range(4)]
        gameplays = [NortheastGameplay(i) for i in range(4)]

        current_kyoku = 0
        current_scores = [0, 0, 0, 0]

        for ev in events:
            t = ev.get("type", "")

            if t == "start_game":
                current_scores = [0, 0, 0, 0]

            elif t == "start_kyoku":
                current_scores = ev.get("scores", [0, 0, 0, 0])[:]
                for gp in gameplays:
                    gp._current_kyoku = current_kyoku

            elif t == "end_kyoku":
                # Mark last record of each gameplay as done=True
                for gp in gameplays:
                    if gp._dones:
                        gp._dones[-1] = True
                current_kyoku += 1

            elif t == "end_game":
                for gp in gameplays:
                    gp._grp.set_final(current_scores)
                break

            elif t == "hora":
                deltas = ev.get("deltas", [0, 0, 0, 0])
                for i in range(4):
                    current_scores[i] += deltas[i]
                for gp in gameplays:
                    gp._grp.add_kyoku(deltas)

            elif t == "ryukyoku":
                deltas = ev.get("deltas", [0, 0, 0, 0])
                for i in range(4):
                    current_scores[i] += deltas[i]
                for gp in gameplays:
                    gp._grp.add_kyoku(deltas)

            # Feed event to all four player states
            for player_id in range(4):
                result = states[player_id].update(ev)
                if result is not None:
                    obs, action, mask = result
                    gp = gameplays[player_id]
                    gp.add_record(
                        obs=obs,
                        action=action,
                        mask=mask,
                        at_turn=states[player_id].at_turn,
                        shanten=states[player_id].shanten,
                        done=False,
                    )

        return [gp for gp in gameplays if gp._obs]
