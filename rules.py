import numpy


def rule_180_with_signals(matrix, agent, neighborhood):
    """
    Returns new y coord of the agent, with respect to elementary cellular automata rule 180
    and existence of traffic signals (agents with special state -1)

    Args:
        matrix: state of all system in matrix form
        agent: tuple (state, x, y) which contains state of the agent and his coords in matrix. x - lane number, y -
        square number in lane
        neighborhood: list of all other agents in agent's neighborhood
    Returns:
        new state of the agent in same (state, x, y) form
    """

    forward_agent = next((a for a in neighborhood if a[1] == agent[1] and a[2] == agent[2] + 1), None)
    if forward_agent is None:
        #traffic signal is red
        if agent[0] == -1:
            return agent
        else:
            return agent[0], agent[1], agent[2] + 1
    else:
            return agent
