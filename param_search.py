from sklearn.model_selection import ParameterGrid
from datetime import datetime
import os
import pickle
import json
import pandas as pd

from src.train import train_nn_agent, train_value_table_agent
from src.eval import evaluate_agents
from src.env import Env


def save_agent_and_config(directory_path, agent, config):
    # create new subdir for each agent
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_directory = os.path.join(directory_path, current_time)
    os.makedirs(new_directory, exist_ok=True)

    # save agent as pickle
    agent_filename = os.path.join(new_directory, 'agent.pkl')
    with open(agent_filename, 'wb') as agent_file:
        pickle.dump(agent, agent_file)

    # save config as json
    config_filename = os.path.join(new_directory, 'config.json')
    with open(config_filename, 'w') as config_file:
        json.dump(config, config_file, indent=4)


def load_agents_and_configs(directory_path="artifacts/agents"):
    agents_and_configs = []

    # iterate over all subdirectories
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)
        if os.path.isdir(subdir_path):
            # load agent from pickle
            agent_filename = os.path.join(subdir_path, 'agent.pkl')
            with open(agent_filename, 'rb') as agent_file:
                agent = pickle.load(agent_file)

            # load config from json
            config_filename = os.path.join(subdir_path, 'config.json')
            with open(config_filename, 'r') as config_file:
                config = json.load(config_file)

            agents_and_configs.append((agent, config))

    return agents_and_configs


def search_agents(save_dir="artifacts/agents"):
    # train NN agents
    fixed_params = {
        'TRAIN_GAMES': [10],
        'APPRENTICE_UPGRADE_CNT': [30],
        'EVALUATE_EVERY_STEP': [10], 
        'EVALUATION_GAMES': [10],
        'BEST_NET_WIN_RATIO': [0.7],

        'PATIENCE_MAX': [5],
        'VAL_SET_SIZE': [1024],
    }

    params = {
        'BATCH_SIZE': [32, 64, 128],

        'VALUE_TABLE_ALPHA': [0.1],

        'EPS': [1], 
        'EPS_MIN': [0.1, 0.01], 
        'EPS_STEP': [0.005], 

        'REPLAY_BUFFER_LEN': [10000, 20000], 
        'PRIO_BUFFER_ALPHA': [0.6, 0.9], 
        'PRIO_BUFFER_BETA': [0.5], 
        'PRIO_BUFFER_BETA_STEP': [0.005, 0.05],

        'BUFFER_TYPE': ['regular', 'prio']
    }
    params.update(fixed_params)
    grid = ParameterGrid(params)

    for config in grid:
        agent, _ = train_nn_agent(config)
        config['AGENT_TYPE'] = 'NN'
        save_agent_and_config(save_dir, agent, config)
        print("Agent saved")

    # train Value Table agents
    params = {
        'TRAIN_GAMES': [10],
        'APPRENTICE_UPGRADE_CNT': [100],
        'EVALUATE_EVERY_STEP': [10], 
        'EVALUATION_GAMES': [10],
        'BEST_NET_WIN_RATIO': [0.7],

        'EPS': [1], 
        'EPS_MIN': [0.1, 0.01], 
        'EPS_STEP': [0.005, 0.01], 

        'VALUE_TABLE_ALPHA': [0.05, 0.1, 0.2]
    }
    grid = ParameterGrid(params)

    for config in grid:
        agent = train_value_table_agent(config)
        config['AGENT_TYPE'] = 'ValueTable'
        save_agent_and_config(save_dir, agent, config)
        print("Agent saved")


class ScoredAgent:
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.score  = 0

def swiss_tournament(scored_agents, rounds, save_path=None):
    env = Env()

    for round_num in range(rounds):
        # Sort agents by score
        scored_agents.sort(key=lambda x: x.score, reverse=True)
        print(f"Round {round_num + 1}")
        for i in range(0, len(scored_agents) - 1, 2):
            scored_agent1, scored_agent2 = scored_agents[i], scored_agents[i + 1]
            result = evaluate_agents(scored_agent1.agent, scored_agent2.agent, env, 100, randomness=False) # winrate
            if result > 0.5:
                scored_agent1.score += 1
            elif result < 0.5:
                scored_agent2.score += 1
            # no score change for a draw
    scored_agents.sort(key=lambda x: x.score, reverse=True)

    if save_path:
        res_entries = []
        for scored_agent in scored_agents:
            res_agent = scored_agent.config
            res_agent['score'] = scored_agent.score
            res_entries.append(res_agent)
        df_res = pd.DataFrame.from_records(res_entries)
        df_res.to_excel(os.path.join(save_path, "scores.xlsx"))

    return scored_agents