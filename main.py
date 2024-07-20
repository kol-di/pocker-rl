from param_search import (
    search_agents, 
    load_agents_and_configs, 
    swiss_tournament, 
    ScoredAgent
)


def main():
    # train ans save agents with different hyperparams
    search_agents(save_dir="artifacts/agents")

    # play tournament between agents to find the best one
    scored_agents = [ScoredAgent(agent, config) for agent, config in load_agents_and_configs(directory_path="artifacts/agents")]
    swiss_tournament(scored_agents, rounds=15, save_path="artifacts/scores")


if __name__ == '__main__':
    main()