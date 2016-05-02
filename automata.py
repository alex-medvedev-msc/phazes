import numpy


def rule_180(matrix, agent, neighborhood):
    """
    Returns new y coord of the agent, with respect to elementary cellular automata rule 180

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
        return agent[0], agent[1], agent[2] + 1
    else:
        return agent


def apply_rule(matrix, new_matrix, agent, neighborhood, rule):
    """
    This function will apply rule to agent with respect to his neighborhood in matrix, and WILL CHANGE new_matrix

    Args:
        matrix: current state of all system in matrix form
        new_matrix: future state of all system in matrix form
        agent: tuple (state, x, y) which contains state of the agent and his coords in matrix. x - lane number, y -
        square number in lane
        neighborhood: list of all other agents in agent's neighborhood
        rule: function object, which takes three params (matrix, agent, neighborhood) and returns new state and coords
        of the agent in tuple (state, x, y) form
    Raises:
        ValueError: New lane number 1 of agent (10,1,100) was out of road bounds
        ValueError: new_matrix state of 0, 0 coords is not zero
    """

    state, x, y = rule(matrix, agent, neighborhood)
    if x < 0 or x >= matrix.shape[0]:
        raise ValueError("New lane number {1} of agent {0} was out of road bounds".format(agent, x))
    elif 0 <= y < matrix.shape[1]:
        if new_matrix[x, y]:
            raise ValueError("new_matrix state of {0}, {1} coords is not zero".format(x, y))
        new_matrix[x, y] = state


def full_neighborhood(matrix, agent):
    """
    Returns list of agents, which can be possibly considered as neighbors by any rule

    Args:
        matrix: current state of all system in matrix form
        agent: tuple (state, x, y) which contains state of the agent and his coords in matrix. x - lane number, y -
        square number in lane
    Returns:
        List of agents [(state, x, y)]
    """

    state, x, y = agent
    rows, cols = matrix.shape
    neighborhood = []
    if y + 1 < cols:
        forward = find_agent(matrix[x, y + 1:], False)
        if forward is not None:
            x_new, y_new = x, y + 1 + forward[1]
            neighborhood.append((matrix[x_new, y_new], x_new, y_new))
    if y - 1 >= 0:
        back = find_agent(matrix[x, :y], True)
        if back is not None:
            x_new, y_new = x, y - 1 - back[1]
            neighborhood.append((matrix[x_new, y_new], x_new, y_new))
    if x + 1 < rows:
        forward = find_agent(matrix[x + 1, y:], False)
        if forward is not None:
            x_new, y_new = x + 1, y + forward[1]
            neighborhood.append((matrix[x_new, y_new], x_new, y_new))
    if x + 1 < rows:
        back = find_agent(matrix[x + 1, :y + 1], True)
        if back is not None:
            x_new, y_new = x + 1, y - back[1]
            neighborhood.append((matrix[x_new, y_new], x_new, y_new))
    if x > 0:
        forward = find_agent(matrix[x - 1, y:], False)
        if forward is not None:
            x_new, y_new = x - 1, y + forward[1]
            neighborhood.append((matrix[x_new, y_new], x_new, y_new))
    if x > 0:
        back = find_agent(matrix[x - 1, :y + 1], True)
        if back is not None:
            x_new, y_new = x - 1, y - back[1]
            neighborhood.append((matrix[x_new, y_new], x_new, y_new))
    # To fix a weird situation, when two agents from lanes 0 and 2 can simultaneously change lane to 1
    # and collide in same square
    if x + 2 < rows:
        back = find_agent(matrix[x + 2, :y + 1], True)
        if back is not None:
            x_new, y_new = x + 2, y - back[1]
            neighborhood.append((matrix[x_new, y_new], x_new, y_new))
    if x - 2 >= 0:
        back = find_agent(matrix[x - 2, :y + 1], True)
        if back is not None:
            x_new, y_new = x - 2, y - back[1]
            neighborhood.append((matrix[x_new, y_new], x_new, y_new))

    return list(set(neighborhood))


def find_agent(lane, reversed=False):
    """
    Returns first nonzero element and its index in one-dimensional array

    Args:
        lane: one-dimensional numpy array
        reversed: if True, then search will start from end of lane
    Returns:
        (element, index) tuple or None if there is no nonzero element in array
    """
    if reversed:
        indices = numpy.nonzero(lane[::-1])
    else:
        indices = numpy.nonzero(lane)
    if len(indices[0]) > 0:
        return lane[indices[0][0]], indices[0][0]
    else:
        return None


def iterate(matrix, rule):
    """
    Returns new matrix (state of all system) after applying to old matrix specific rule

    Args:
        matrix: current state of all system in matrix form
        rule: function object, which takes three params (matrix, agent, neighborhood) and returns new state and coords
        of the agent in tuple (state, x, y) form
    Returns:
        two-dimensional numpy array of same shape as matrix and
    """

    new_matrix = numpy.zeros(matrix.shape, dtype=matrix.dtype)
    for lane in numpy.arange(matrix.shape[0]):
        for square in numpy.arange(matrix.shape[1]):
            if matrix[lane, square] == 0:
                continue
            agent = (matrix[lane, square], lane, square)
            neighborhood = full_neighborhood(matrix, agent)
            apply_rule(matrix, new_matrix, agent, neighborhood, rule)

    return new_matrix


def simulate(start=None, lanes=1, length=100, steps=10, rule=rule_180, monitor=None):
    """
    Run specified number of steps by applying rule to whole matrix

    Args:
        start: starting state of the whole system, it will be zero-filled if start is None
        lanes: number of lanes on road if start is None
        length: number of squares in each lane if start is None
        steps: number of iterations of simulation
        rule: function object, which takes three params (matrix, agent, neighborhood) and returns new state and coords
        of the agent in tuple (state, x, y) form
        monitor: monitor object with func observe(new_matrix, old_matrix).
        It cannot change the matrices, otherwise consequences will be awful
    Returns:
        final state of the system in the matrix form with the same dimensions as original start matrix, or (lanes, length)
    """

    if start is None:
        start = numpy.zeros((lanes, length), dtype=int)

    old_matrix = start
    for step in range(steps):
        new_matrix = iterate(old_matrix, rule)
        if monitor is not None:
            monitor.observe(new_matrix, old_matrix)
        old_matrix = new_matrix
    return old_matrix
