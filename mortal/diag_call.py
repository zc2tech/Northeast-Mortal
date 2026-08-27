"""
Diagnostic: load a checkpoint and show Q-values for call-decision samples.
Run from mortal/ directory:
    python3 diag_call.py

Loads the first few call-decision observations from training data and prints
Q-values for PASS, PON, CHI_LOW, CHI_MID, CHI_HIGH so you can see if the
model prefers calling or passing when it has the opportunity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from glob import glob
from config import config
from model import Brain, DQN
from libne.dataset import NortheastGameplayLoader
from libne.consts import (
    ACTION_PASS, ACTION_PON, ACTION_CHI_LOW, ACTION_CHI_MID, ACTION_CHI_HIGH, ACTION_KAN, ACTION_AGARI
)

CALL_ACTIONS = {
    'CHI_LOW': ACTION_CHI_LOW,
    'CHI_MID': ACTION_CHI_MID,
    'CHI_HIGH': ACTION_CHI_HIGH,
    'PON':     ACTION_PON,
    'KAN':     ACTION_KAN,
    'AGARI':   ACTION_AGARI,
    'PASS':    ACTION_PASS,
}

def main():
    device = torch.device('cpu')
    state_file = config['control']['state_file']
    if not os.path.exists(state_file):
        print(f'No checkpoint found at {state_file}')
        return

    state = torch.load(state_file, weights_only=True, map_location=device)
    cfg = state['config']
    version = cfg['control'].get('version', 4)
    mortal = Brain(version=version, **cfg['resnet']).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    print(f'Loaded checkpoint: step={state["steps"]}, version={version}')

    # Load a few game files
    files = []
    for pat in config['dataset']['globs']:
        files.extend(glob(pat, recursive=True))
    files = files[:5]
    if not files:
        print('No game files found')
        return

    loader = NortheastGameplayLoader()
    gameplays = loader.load_gz_log_files(files)

    call_samples = []  # (obs, mask, action_taken)
    for gp in gameplays:
        obs_list = gp.take_obs()
        actions = gp.take_actions()
        masks = gp.take_masks()
        for obs, action, mask in zip(obs_list, actions, masks):
            # Call decision: mask has at least one of chi/pon/kan/agari AND pass
            has_call = any(mask[a] for a in [ACTION_CHI_LOW, ACTION_CHI_MID, ACTION_CHI_HIGH, ACTION_PON, ACTION_KAN, ACTION_AGARI])
            if has_call and mask[ACTION_PASS]:
                call_samples.append((obs, mask, action))
        if len(call_samples) >= 20:
            break

    if not call_samples:
        print('No call-decision samples found in these files')
        return

    print(f'\nFound {len(call_samples)} call-decision samples. Q-values:\n')
    print(f'{"#":<4} {"taken":<10} {"CHI_L":>7} {"CHI_M":>7} {"CHI_H":>7} {"PON":>7} {"KAN":>7} {"AGARI":>7} {"PASS":>7} {"choice":>10}')
    print('-' * 80)

    with torch.inference_mode():
        for i, (obs, mask, action_taken) in enumerate(call_samples[:20]):
            obs_t = torch.as_tensor(obs[np.newaxis], dtype=torch.float32)
            mask_t = torch.as_tensor(mask[np.newaxis], dtype=torch.bool)
            phi = mortal(obs_t)
            q_out = dqn(phi, mask_t)[0]  # (ACTION_SPACE,)

            q = {name: q_out[idx].item() if mask[idx] else float('nan')
                 for name, idx in CALL_ACTIONS.items()}

            taken_name = next((n for n, idx in CALL_ACTIONS.items() if idx == action_taken), str(action_taken))
            # model's choice among legal actions
            legal_q = {n: v for n, v in q.items() if not np.isnan(v)}
            model_choice = max(legal_q, key=legal_q.get)

            print(f'{i:<4} {taken_name:<10} '
                  f'{q["CHI_LOW"]:>7.2f} {q["CHI_MID"]:>7.2f} {q["CHI_HIGH"]:>7.2f} '
                  f'{q["PON"]:>7.2f} {q["KAN"]:>7.2f} {q["AGARI"]:>7.2f} '
                  f'{q["PASS"]:>7.2f} {model_choice:>10}')

    # Summary
    model_choices = []
    with torch.inference_mode():
        for obs, mask, _ in call_samples:
            obs_t = torch.as_tensor(obs[np.newaxis], dtype=torch.float32)
            mask_t = torch.as_tensor(mask[np.newaxis], dtype=torch.bool)
            phi = mortal(obs_t)
            q_out = dqn(phi, mask_t)[0]
            legal = {n: q_out[idx].item() for n, idx in CALL_ACTIONS.items() if mask[idx]}
            model_choices.append(max(legal, key=legal.get))

    from collections import Counter
    counts = Counter(model_choices)
    total = len(model_choices)
    print(f'\nModel choice distribution over {total} call-decision samples:')
    for name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f'  {name:<10} {cnt:>4} ({100*cnt/total:.1f}%)')

if __name__ == '__main__':
    main()
