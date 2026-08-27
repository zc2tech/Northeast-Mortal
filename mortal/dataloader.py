import random
import torch
import numpy as np
from torch.utils.data import IterableDataset
from libne.dataset import NortheastGameplayLoader
from config import config

class FileDatasetsIter(IterableDataset):
    def __init__(
        self,
        version,
        file_list,
        pts,
        oracle = False,
        file_batch_size = 20,
        reserve_ratio = 0,
        player_names = None,
        excludes = None,
        num_data_passes = 1,
    ):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.pts = pts
        self.oracle = oracle
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        self.player_names = player_names
        self.excludes = excludes
        self.num_data_passes = num_data_passes
        self.iterator = None

    def build_iter(self):
        self.loader = NortheastGameplayLoader()
        for _ in range(self.num_data_passes):
            yield from self.load_files()

    def load_files(self):
        random.shuffle(self.file_list)
        self.buffer = []

        for start_idx in range(0, len(self.file_list), self.file_batch_size):
            old_buffer_size = len(self.buffer)
            self.populate_buffer(self.file_list[start_idx:start_idx + self.file_batch_size])
            buffer_size = len(self.buffer)

            reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
            if reserved_size > buffer_size:
                continue

            random.shuffle(self.buffer)
            yield from self.buffer[reserved_size:]
            del self.buffer[reserved_size:]
        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def populate_buffer(self, file_list):
        # NortheastGameplayLoader returns a flat list of NortheastGameplay objects
        gameplays = self.loader.load_gz_log_files(file_list)
        for game in gameplays:
            obs = game.take_obs()
            actions = game.take_actions()
            masks = game.take_masks()
            at_kyoku = game.take_at_kyoku()
            dones = game.take_dones()
            apply_gamma = game.take_apply_gamma()

            grp = game.take_grp()
            player_id = game.take_player_id()

            game_size = len(obs)
            if game_size == 0:
                continue

            # Compute per-kyoku rewards from logged score deltas.
            grp_feature = grp.take_feature()   # (N_kyoku, 4) float32
            kyoku_rewards_raw = grp_feature[:, player_id]  # (N_kyoku,) deltas normalised by /100
            kyoku_rewards = kyoku_rewards_raw

            max_kyoku = int(at_kyoku[-1]) + 1 if at_kyoku else 1
            if len(kyoku_rewards) < max_kyoku:
                # Pad with zeros if fewer kyoku rewards than expected
                pad = np.zeros(max_kyoku - len(kyoku_rewards), dtype=np.float32)
                kyoku_rewards = np.concatenate([kyoku_rewards, pad])

            # Compute final rank sequence for aux next-rank prediction
            final_scores = grp.take_final_scores()
            # Build rank at each kyoku boundary (cumulative scores)
            n_kyoku = len(grp_feature)
            cum_scores = np.cumsum(grp_feature * 30000.0, axis=0)  # (N_kyoku, 4)
            # Prepend zeros row so rank_seq has n_kyoku+1 rows
            scores_seq = np.vstack([np.zeros((1, 4)), cum_scores])
            rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
            # player_ranks[k] = rank of this player at start of kyoku k
            player_ranks_arr = rank_by_player_seq[:, player_id]  # (N_kyoku+1,)

            # steps_to_done: distance to end of kyoku
            steps_to_done = np.zeros(game_size, dtype=np.int64)
            for i in reversed(range(game_size)):
                if not dones[i]:
                    steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

            for i in range(game_size):
                kyoku_idx = int(at_kyoku[i])
                reward_idx = min(kyoku_idx, len(kyoku_rewards) - 1)
                rank_idx = min(kyoku_idx + 1, len(player_ranks_arr) - 1)
                entry = [
                    obs[i],
                    actions[i],
                    masks[i],
                    steps_to_done[i],
                    kyoku_rewards[reward_idx],
                    player_ranks_arr[rank_idx],
                ]
                self.buffer.append(entry)

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator

def worker_init_fn(*args, **kwargs):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    if not isinstance(dataset, FileDatasetsIter):
        return
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]
