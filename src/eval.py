import numpy as np
import pickle
import matplotlib.pyplot as plt
from IPython.display import clear_output
from itertools import product
import seaborn as sns

from src.game import play_game
from src.env import round_state


def compare_agents(agent1, agent2, env, n_games):
    n1_win, n2_win = 0, 0

    for _ in range(n_games):
        _, winner = play_game(agent1, agent2, env)
        if winner == 0:
            n1_win += 1
        else:
            n2_win += 1

    return n1_win / (n1_win + n2_win)


def evaluate_agents(agent1, agent2, env, n_games, randomness=True):
    if isinstance(agent1, str):
        with open(agent1, 'rb') as f:
            agent1 = pickle.load(f)
    if isinstance(agent2, str):
        with open(agent2, 'rb') as f:
            agent2 = pickle.load(f)

    if not randomness:
        with agent1.eval(), agent2.eval():
            return compare_agents(agent1, agent2, env, n_games)
        
    return compare_agents(agent1, agent2, env, n_games)


def monitor(train_loss, val_loss, mean_game_rounds, l2_grads, upgrade_iter=False):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.title('Train loss')
    plt.plot(train_loss)

    plt.subplot(2, 2, 2)
    plt.title('Val loss')
    plt.plot(val_loss)

    plt.subplot(2, 2, 3)
    plt.title('Mean rounds in a game')
    plt.plot(mean_game_rounds)
    if upgrade_iter:
        for upg in upgrade_iter:
            plt.axvline(upg, color='black', linestyle='--')

    plt.subplot(2, 2, 4)
    plt.title('Grad L2')
    plt.plot(l2_grads)
    if upgrade_iter:
        for upg in upgrade_iter:
            plt.axvline(upg, color='black', linestyle='--')

    plt.tight_layout()
    plt.show()
    clear_output(wait=True);



def plot_learned_rewards(agent):
    if isinstance(agent, str):
        with open(agent, 'rb') as f:
            agent = pickle.load(f)

    fig, axes = plt.subplots(13, 2, figsize=(10, 50))
    axes = np.ravel(axes)

    pred_stacks = list(product(range(0, 50, 2), range(0, 50, 2)))
    ranks_blinds = list(product(range(13), (0, 1)))
    with agent.eval():
        for ax, rk_blind in zip(axes, ranks_blinds):
            pred_states = [round_state(sb_stack=sb, bb_stack=bb, blind_order=rk_blind[1], card_rk=rk_blind[0]) for sb, bb in pred_stacks]

            vals_acts = [agent.best_value_and_action(state) for state in pred_states]
            push_preds = np.array([val if act else np.nan for val, act in vals_acts]).reshape(25, 25)
            fold_preds = np.array([val if not act else np.nan for val, act in vals_acts]).reshape(25, 25)
            
            ax.set_title(f"Card rank {rk_blind[0]}, {'BB' if rk_blind[1] else 'SB'}")

            if not (np.isnan(push_preds).all()):
                sns.heatmap(push_preds, ax=ax, cmap='crest', mask=(push_preds == None))
            if not (np.isnan(fold_preds).all()):
                sns.heatmap(fold_preds, ax=ax, cmap='flare', mask=(fold_preds == None))

            ax.set_xticks(np.linspace(0, 25, 11))
            ax.set_yticks(np.linspace(0, 25, 11))
            ax.set_xticklabels(np.linspace(0, 50, 11).astype(int))
            ax.set_yticklabels(np.linspace(0, 50, 11).astype(int))

            ax.set_xlabel("BB stack")
            ax.set_ylabel("SB stack")

    plt.tight_layout()