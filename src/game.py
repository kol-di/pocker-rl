from numpy.random import choice

from src.env import round_state


SMALL_BLIND = 1
BIG_BLIND = 2


def play_round(agent_sb, agent_bb, common_state):

    # randomly choose card ranke for SB with uniform rank distribution
    card_rk_sb = choice(range(13))

    # probability to choose same rank as SB is smaller for BB
    bb_rk_probs = [4/51] * 13
    bb_rk_probs[card_rk_sb] = 3/51
    card_rk_bb = choice(range(13), p=bb_rk_probs)

    # each agent's info is limited, so their states differ
    state_sb = common_state._replace(blind_order=0, card_rk=card_rk_sb)
    state_bb = common_state._replace(blind_order=1, card_rk=card_rk_bb)

    # SB moves
    _, action_sb = agent_sb.best_value_and_action(state_sb)
    if action_sb == 0: # fold
        r_sb = -SMALL_BLIND
        r_bb = SMALL_BLIND

        # if SB folded we only use his experience, since for BB no choice was made
        exp = [(state_sb, action_sb, r_sb)]

    # BB moves
    else:
        _, action_bb = agent_bb.best_value_and_action(state_bb)
        if action_bb == 1: # push
            if card_rk_sb > card_rk_bb:
                r_sb = min(common_state.sb_stack, common_state.bb_stack)
                r_bb = -r_sb
            elif card_rk_sb < card_rk_bb:
                r_bb = min(common_state.sb_stack, common_state.bb_stack)
                r_sb = -r_bb
            else:
                r_sb, r_bb = 0, 0
        elif action_bb == 0: # fold
            r_sb = BIG_BLIND
            r_bb = -BIG_BLIND

        exp = [(state_sb, action_sb, r_sb), (state_bb, action_bb, r_bb)]

    # in the next round players swap blind order and stacks
    next_round_state = round_state(
        bb_stack=common_state.sb_stack + r_sb,
        sb_stack=common_state.bb_stack + r_bb,
        blind_order=(common_state.blind_order + 1) % 2
    )

    return exp, next_round_state


def play_game(agent1, agent2, env):
    exp = []

    state = None
    next_state = env.random_start_state()

    agents = [agent1, agent2]
    agent1_is_sb = True

    while next_state.sb_stack > SMALL_BLIND and next_state.bb_stack > BIG_BLIND:
        state = next_state

        agents = list(reversed(agents))
        agent_sb, agent_bb = agents
        agent1_is_sb = not agent1_is_sb

        round_exp, next_state = play_round(agent_sb, agent_bb, state)
        exp.extend(round_exp)

    # return index of winner
    winner = None
    if next_state.sb_stack <= SMALL_BLIND:
        if agent1_is_sb:
            winner = 0
        else:
            winner = 1
    if next_state.bb_stack <= BIG_BLIND:
        if agent1_is_sb:
            winner = 1
        else:
            winner = 0

    return exp, winner