import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn

from src.game import play_game
from src.eval import compare_agents
from src.env import Env
from src.agents import ValueTableAgent, NNAgent
from src.util.replay_buffer import ReplayBuffer, PrioReplayBuffer
from src.util.loss import WeightedMSE



def train_nn_agent(config):

    buffer_type = config['BUFFER_TYPE']

    if buffer_type == 'regular':
        buffer = ReplayBuffer(config)
        loss = nn.MSELoss()
    elif buffer_type == 'prio':
        buffer = PrioReplayBuffer(config)
        loss = WeightedMSE()
    else:
        raise Exception("Unknown buffer_type")
    
    best_agent = NNAgent(config, loss)
    apprentice_agent = NNAgent(config, loss)

    mean_game_rounds = []
    l2_grads = []
    train_loss_history = []
    val_loss_history = []
    upgrade_iters = []
    env = Env()


    val_set_size = config['VAL_SET_SIZE']

    # fill the replay buffer
    while not buffer.full():
        for _ in range(config['TRAIN_GAMES']):
            buffer.extend( play_game(best_agent, best_agent, env)[0] )
    buffer_accum_before = buffer.len()

    # create synthetic validation set to monitor for early stopping
    if buffer_type == 'regular':
        val_sample = buffer.sample(val_set_size)
    elif buffer_type == 'prio':
        val_sample, _, _ = buffer.sample(val_set_size)
    best_val_loss = np.iinfo(np.int32).max
    patience_cnt = 0
    best_apprentice_agent = None


    iter_no = 0
    apprentice_upgrade_cnt = 0
    while apprentice_upgrade_cnt < config['APPRENTICE_UPGRADE_CNT']:
        iter_no += 1

        for _ in range(config['TRAIN_GAMES']):
            buffer.extend( play_game(best_agent, best_agent, env)[0] )

        apprentice_agent.decrease_eps()

        if buffer_type == 'regular':
            # sample randomly from buffer and update network weights
            batch = buffer.sample(config['BATCH_SIZE'])
            iter_grads, train_loss = apprentice_agent.value_update(batch)
        elif buffer_type == 'prio':
            # update weight-normalizing parameter
            buffer.update_beta()

            # update network using weighted samples
            batch, batch_indices, batch_weights = buffer.sample(config['BATCH_SIZE'])
            iter_grads, train_loss = apprentice_agent.value_update(batch, batch_weights)
            
            # update sample priorities using loss for corresponding samples
            batch_prios = apprentice_agent.loss.sample_prios
            buffer.update_priorities(batch_indices, batch_prios)

        if iter_no % config['EVALUATE_EVERY_STEP'] == 0:
            win_ratio = compare_agents(apprentice_agent, best_agent, env, n_games=config['EVALUATION_GAMES'])
            print("Net evaluated, win ratio = %.2f" % win_ratio)

            if win_ratio >= config['BEST_NET_WIN_RATIO']:
                print(f"Net is better than cur best, sync on iter {iter_no}")
                best_agent = deepcopy(apprentice_agent)
                apprentice_upgrade_cnt += 1
                upgrade_iters.append(iter_no)

                # monitor validation loss
                with torch.no_grad():
                    val_loss = apprentice_agent.calc_loss(val_sample).item()
                val_loss_history.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_apprentice_agent = deepcopy(apprentice_agent)
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= config['PATIENCE_MAX']:
                        print(f'Triggered early stopping on iter {iter_no}')
                        apprentice_agent = best_apprentice_agent
                        break


        ####################### Monitor #####################################

        # loss
        train_loss_history.append(train_loss)

        # calculate mean number of rounds a single game took
        buffer_accum_after = buffer.accum_len
        mean_game_rounds.append( (buffer_accum_after-buffer_accum_before)/config['TRAIN_GAMES'] )
        buffer_accum_before = buffer_accum_after

        # log results
        l2_grads.append( np.sqrt(np.mean(np.square(iter_grads))) )

    stats = {
        "train_loss": train_loss_history, 
        "val_loss": val_loss_history, 
        "mean_game_rounds": mean_game_rounds, 
        "l2_grads": l2_grads
    }

    return apprentice_agent, stats



def train_value_table_agent(config):
    env = Env()

    best_agent = ValueTableAgent(config)
    apprentice_agent = ValueTableAgent(config)

    iter_no = 0
    apprentice_upgrade_cnt = 0
    while apprentice_upgrade_cnt < config['APPRENTICE_UPGRADE_CNT']:
        iter_no += 1

        exp = []
        for _ in range(config['TRAIN_GAMES']):
            exp.extend( play_game(best_agent, best_agent, env)[0] )

        for exp_sample in exp:
            apprentice_agent.value_update(*exp_sample)

        if iter_no % config['EVALUATE_EVERY_STEP'] == 0:
            win_ratio = compare_agents(apprentice_agent, best_agent, env, n_games=config['EVALUATION_GAMES'])
            print("Net evaluated, win ratio = %.2f" % win_ratio)

            if win_ratio >= config['BEST_NET_WIN_RATIO']:
                print("Net is better than cur best, sync")
                apprentice_agent.decrease_eps()
                best_agent = deepcopy(apprentice_agent)
                apprentice_upgrade_cnt += 1

    return apprentice_agent
