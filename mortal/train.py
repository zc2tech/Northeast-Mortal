def train():
    import prelude

    import logging
    import sys
    import os
    import gc
    import gzip
    import json
    import random
    import torch
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from common import submit_param, parameter_count, drain, filtered_trimmed_lines, tqdm
    from dataloader import FileDatasetsIter, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, DQN, AuxNet
    from ne_consts import obs_shape
    from config import config

    version = config['control']['version']

    online = config['control']['online']
    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    submit_every = config['control']['submit_every']
    min_q_weight = config['cql']['min_q_weight']
    next_rank_weight = config['aux']['next_rank_weight']
    assert save_every % opt_step_every == 0

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']

    pts = config['env']['pts']
    gamma = config['env']['gamma']
    file_batch_size = config['dataset']['file_batch_size']
    reserve_ratio = config['dataset']['reserve_ratio']
    num_workers = config['dataset']['num_workers']
    num_data_passes = config['dataset']['num_data_passes']
    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']
    max_grad_norm = config['optim']['max_grad_norm']

    mortal = Brain(version=version, **config['resnet']).to(device)
    dqn = DQN(version=version).to(device)
    aux_net = AuxNet((4,)).to(device)
    all_models = (mortal, dqn, aux_net)
    if enable_compile:
        for m in all_models:
            m.compile()

    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'dqn params: {parameter_count(dqn):,}')
    logging.info(f'aux params: {parameter_count(aux_net):,}')

    mortal.freeze_bn(config['freeze_bn']['mortal'])

    decay_params = []
    no_decay_params = []
    for model in all_models:
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
    ]
    optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps)
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    scaler = GradScaler(device.type, enabled=enable_amp)

    steps = 0
    state_file = config['control']['state_file']

    if not online and path.exists(state_file):
        file_index = config['dataset']['file_index']
        current_files = []
        for pat in config['dataset']['globs']:
            current_files.extend(glob(pat, recursive=True))
        if not current_files:
            logging.warning('no dataset files found — deleting stale state and file index to restart training from scratch')
            os.remove(state_file)
            if path.exists(file_index):
                os.remove(file_index)

    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        aux_net.load_state_dict(state['aux_net'])
        if not online or state['config']['control']['online']:
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        steps = state['steps']

    optimizer.zero_grad(set_to_none=True)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    if online:
        submit_param(mortal, dqn, is_idle=True)
        logging.info('param has been submitted')

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'dqn_loss': 0,
        'cql_loss': 0,
        'next_rank_loss': 0,
    }
    all_q = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    all_q_target = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    idx = 0

    def train_epoch():
        nonlocal steps
        nonlocal idx

        player_names = []
        if online:
            player_names = ['trainee']
            dirname = drain()
            file_list = list(map(lambda p: path.join(dirname, p), os.listdir(dirname)))
        else:
            player_names_set = set()
            for filename in config['dataset']['player_names_files']:
                with open(filename) as f:
                    player_names_set.update(filtered_trimmed_lines(f))
            player_names = list(player_names_set)
            logging.info(f'loaded {len(player_names):,} players')

            file_index = config['dataset']['file_index']
            # Scan disk to get current file list
            current_files = []
            for pat in config['dataset']['globs']:
                current_files.extend(glob(pat, recursive=True))
            if len(player_names_set) > 0:
                current_files = [f for f in current_files
                                 if not set(json.loads(next(gzip.open(f, 'rt')))[  'names']).isdisjoint(player_names_set)]
            current_files.sort(reverse=True)

            if path.exists(file_index):
                index = torch.load(file_index, weights_only=True)
                cached = index['file_list']
                if set(cached) == set(current_files):
                    file_list = cached
                    logging.info('file index up to date')
                else:
                    added   = len(set(current_files) - set(cached))
                    removed = len(set(cached) - set(current_files))
                    logging.info(f'file index stale (+{added}/-{removed}), rebuilding...')
                    file_list = current_files
                    torch.save({'file_list': file_list}, file_index)
            else:
                logging.info('building file index...')
                file_list = current_files
                torch.save({'file_list': file_list}, file_index)
        logging.info(f'file list size: {len(file_list):,}')

        if num_workers > 1:
            random.shuffle(file_list)
        file_data = FileDatasetsIter(
            version = version,
            file_list = file_list,
            pts = pts,
            file_batch_size = file_batch_size,
            reserve_ratio = reserve_ratio,
            player_names = player_names,
            num_data_passes = num_data_passes,
        )
        data_loader = iter(DataLoader(
            dataset = file_data,
            batch_size = batch_size,
            drop_last = False,
            num_workers = num_workers,
            pin_memory = device.type == 'cuda',
            worker_init_fn = worker_init_fn,
        ))

        remaining_obs = []
        remaining_actions = []
        remaining_masks = []
        remaining_steps_to_done = []
        remaining_kyoku_rewards = []
        remaining_player_ranks = []
        remaining_bs = 0
        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)

        def train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks):
            nonlocal steps
            nonlocal idx
            nonlocal pb

            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            steps_to_done = steps_to_done.to(dtype=torch.int64, device=device)
            kyoku_rewards = kyoku_rewards.to(dtype=torch.float32, device=device)
            player_ranks = player_ranks.to(dtype=torch.int64, device=device)
            assert masks[range(batch_size), actions].all(), \
                f'action mask violation at step {steps}'

            q_target_mc = gamma ** steps_to_done * kyoku_rewards
            q_target_mc = q_target_mc.to(torch.float32)

            with torch.autocast(device.type, enabled=enable_amp):
                phi = mortal(obs)
                q_out = dqn(phi, masks)
                q = q_out[range(batch_size), actions]
                dqn_loss = 0.5 * mse(q, q_target_mc)
                cql_loss = 0
                if not online:
                    cql_loss = q_out.logsumexp(-1).mean() - q.mean()

                next_rank_logits, = aux_net(phi)
                next_rank_loss = ce(next_rank_logits, player_ranks)

                loss = sum((
                    dqn_loss,
                    cql_loss * min_q_weight,
                    next_rank_loss * next_rank_weight,
                ))
            scaler.scale(loss / opt_step_every).backward()

            with torch.inference_mode():
                stats['dqn_loss'] += dqn_loss
                if not online:
                    stats['cql_loss'] += cql_loss
                stats['next_rank_loss'] += next_rank_loss
                all_q[idx] = q
                all_q_target[idx] = q_target_mc

            steps += 1
            idx += 1
            if idx % opt_step_every == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                    clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            pb.update(1)

            if online and steps % submit_every == 0:
                submit_param(mortal, dqn, is_idle=False)
                logging.info('param has been submitted')

            if steps % save_every == 0:
                pb.close()

                # downsample to reduce tensorboard event size
                all_q_1d = all_q.cpu().numpy().flatten()[::128]
                all_q_target_1d = all_q_target.cpu().numpy().flatten()[::128]

                writer.add_scalar('loss/dqn_loss', stats['dqn_loss'] / save_every, steps)
                if not online:
                    writer.add_scalar('loss/cql_loss', stats['cql_loss'] / save_every, steps)
                writer.add_scalar('loss/next_rank_loss', stats['next_rank_loss'] / save_every, steps)
                writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
                writer.add_histogram('q_predicted', all_q_1d, steps)
                writer.add_histogram('q_target', all_q_target_1d, steps)
                writer.flush()

                logging.info(f'total steps: {steps:,}, lr: {scheduler.get_last_lr()[0]:.2e}, dqn_loss: {stats["dqn_loss"]/save_every:.4f}, cql_loss: {stats["cql_loss"]/save_every:.4f}')

                for k in stats:
                    stats[k] = 0
                idx = 0

                state = {
                    'mortal': mortal.state_dict(),
                    'current_dqn': dqn.state_dict(),
                    'aux_net': aux_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'steps': steps,
                    'timestamp': datetime.now().timestamp(),
                    'config': config,
                }
                torch.save(state, state_file)

                if online and steps % submit_every != 0:
                    submit_param(mortal, dqn, is_idle=False)
                    logging.info('param has been submitted')

                if online:
                    # BUG: This is a bug with unknown reason. When training
                    # in online mode, the process will get stuck here. This
                    # is the reason why `main` spawns a sub process to train
                    # in online mode instead of going for training directly.
                    sys.exit(0)
                pb = tqdm(total=save_every, desc='TRAIN')

        for obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks in data_loader:
            bs = obs.shape[0]
            if bs != batch_size:
                remaining_obs.append(obs)
                remaining_actions.append(actions)
                remaining_masks.append(masks)
                remaining_steps_to_done.append(steps_to_done)
                remaining_kyoku_rewards.append(kyoku_rewards)
                remaining_player_ranks.append(player_ranks)
                remaining_bs += bs
                continue
            train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks)

        remaining_batches = remaining_bs // batch_size
        if remaining_batches > 0:
            obs = torch.cat(remaining_obs, dim=0)
            actions = torch.cat(remaining_actions, dim=0)
            masks = torch.cat(remaining_masks, dim=0)
            steps_to_done = torch.cat(remaining_steps_to_done, dim=0)
            kyoku_rewards = torch.cat(remaining_kyoku_rewards, dim=0)
            player_ranks = torch.cat(remaining_player_ranks, dim=0)
            start = 0
            end = batch_size
            while end <= remaining_bs:
                train_batch(
                    obs[start:end],
                    actions[start:end],
                    masks[start:end],
                    steps_to_done[start:end],
                    kyoku_rewards[start:end],
                    player_ranks[start:end],
                )
                start = end
                end += batch_size
        pb.close()
        logging.info(f'data exhausted at step {steps:,}, {steps % save_every} unsaved steps in current window')

        if online:
            submit_param(mortal, dqn, is_idle=True)
            logging.info('param has been submitted')

    epoch = 0
    while True:
        epoch += 1
        logging.info(f'epoch {epoch} start, steps so far: {steps:,}')
        try:
            train_epoch()
        except Exception:
            logging.exception(f'train_epoch crashed at step {steps:,}')
            raise
        logging.info(f'epoch {epoch} end, steps so far: {steps:,}')
        gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        if not online:
            # only run one epoch for offline for easier control
            break

def main():
    import argparse
    import os
    import sys
    import time
    from subprocess import Popen
    from config import config

    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', action='store_true', help='delete state file and file index to force training from scratch')
    args = parser.parse_args()

    if args.reset:
        import logging
        from os import path
        state_file = config['control']['state_file']
        file_index = config['dataset']['file_index']
        for f in (state_file, file_index):
            if path.exists(f):
                os.remove(f)
                logging.warning(f'deleted {f}')

    # do not set this env manually
    is_sub_proc_key = 'MORTAL_IS_SUB_PROC'
    online = config['control']['online']
    if not online or os.environ.get(is_sub_proc_key, '0') == '1':
        train()
        return

    cmd = (sys.executable, __file__)
    env = {
        is_sub_proc_key: '1',
        **os.environ.copy(),
    }
    while True:
        child = Popen(
            cmd,
            stdin = sys.stdin,
            stdout = sys.stdout,
            stderr = sys.stderr,
            env = env,
        )
        if (code := child.wait()) != 0:
            sys.exit(code)
        time.sleep(3)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        import logging
        logging.exception('unhandled exception in main')
        raise
