from src.aco_mapf.GraphWorld import *
from src.aco_mapf.GraphWorld import NavigationAgent

import numpy as np

def fitness_proportional_selection(probs, random=np.random.rand()):
    s = sum(probs.values())
    pick = random * s
    current = 0
    for key in sorted(probs):
        current += probs[key]
        if current >= pick:
            return key
    print("selection could not pick an item - should not happen!")
    return probs[-1]

def normalize_matrix(m: np.matrix):
    return m / np.sum(m)


class Colony:
    """
    very simple class to represent the pheromones for multiple ants
    :pheromones matrix with pheromone values
    :ants list with AcoAgents
    """
    pheromones: np.matrix

    def __init__(self, pheromones=None, ant=None):
        self.pheromones = pheromones
        self.ants = []
        if ant is not None:
            self.add_ant(ant)
        self.best_path = None

    def add_ant(self, ant):
        self.ants.append(ant)

    def __len__(self):
        return len(self.ants)

    def vaporize(self, evaporation_method="normalize", evaporation_rate=0.99, **_):
        if evaporation_method == "normalize":
            self.pheromones = normalize_matrix(self.pheromones)
        elif evaporation_method == "default_aco":
            self.pheromones = evaporation_rate ** (1/len(self.ants)) * self.pheromones
        else:
            print(f"evaporation method '{evaporation_method}' does not exist")
            raise NotImplementedError()

    def dissipate(self, dissipation_rate: float = 0.01, **_):
        dissipation_matrix = -1 * self.pheromones * dissipation_rate
        nodes = np.array([0.0 for _ in range(self.pheromones.shape[0])])
        for i in range(self.pheromones.shape[0]):
            nodes[i] = -1 * np.sum(dissipation_matrix[i,:])
        for i in range(self.pheromones.shape[0]):
            neighbours = self.ants[0].world.get_neighbours(i)
            for j in neighbours:
                dissipation_matrix[j, i] += nodes[i] / len(neighbours)
        self.pheromones += dissipation_matrix

class TimeAwareColony(Colony):
    time_pheromones : np.matrix
    time_frame : int
    def __init__(self, **kwargs):
        Colony.__init__(self, **kwargs)
        self.time_frame = []
        self.time_pheromones = None

    def vaporize(self, evaporation_method="normalize", evaporation_rate=0.99, **kwargs):
        Colony.vaporize(self, evaporation_method=evaporation_method, evaporation_rate=evaporation_rate, **kwargs)
        for t in range(self.time_pheromones.shape[0]):
            if evaporation_method == "normalize":
                self.time_pheromones[t] = normalize_matrix(self.time_pheromones[t])
            elif evaporation_method == "default_aco":
                self.time_pheromones[t] = evaporation_rate ** (1/len(self.ants)) * self.time_pheromones[t]
            else:
                print(f"evaporation method '{evaporation_method}' does not exist")
                raise NotImplementedError()

    def dissipate(self, time_dissipation_rate: float = 0.0, **kwargs):
        Colony.dissipate(self, **kwargs)
        for t in range(self.time_pheromones.shape[0]):
            for d in range(max(0, t-1), min(t+1, self.time_pheromones.shape[0])):
                if d != t:
                    self.time_pheromones[t] = self.time_pheromones[t].copy() + time_dissipation_rate * self.time_pheromones[d]


class AcoAgent(NavigationAgent):
    pheromones: np.matrix
    data: list
    random: np.random.RandomState
    best_path: list

    def __init__(self, seed=None, colony=None, **kwargs):
        NavigationAgent.__init__(self, **kwargs)
        self.data = []
        self.path = [self.state]
        self.random = np.random.RandomState(seed=seed)
        self.forward = True
        self.colony = colony if colony is not None else Colony(ant=self)
        self.colony.add_ant(self)
        self.stuck = False
        self.best_path = None
        self.name = "A"
        self.step_count = 0
        self.arrived_counter = 0
        self.stuck_counter = 0
        self._greedy_path = [self.start]
        self._greedy_path_step = -1

    def register_world(self, world):
        NavigationAgent.register_world(self, world)
        self.initialize_pheromones()

    def initialize_pheromones(self):
        if self.colony.pheromones is None:
            self.colony.pheromones = np.ones_like(self.world.adjacency)
            for (i, j), v in np.ndenumerate(self.world.adjacency):
                if v == 0:
                    self.colony.pheromones[i, j] = 0
            self.colony.pheromones = normalize_matrix(self.colony.pheromones)
        else:
            assert self.colony.pheromones.shape == self.world.adjacency.shape

    def transition_value(self, i, j, forward: bool = None, alpha: float = 1.0, beta: float = 1.0, eps: float = 0.01, **_):
        """
        :type alpha: float
        :type beta: float
        """
        if forward is None:
            forward = self.forward
        if not forward:
            i, j = j, i
        return self.colony.pheromones[i, j] ** alpha * (1 / self.world.adjacency[i, j]) ** beta + eps

    def transition_options(self, collision_check : bool = False, own_colony_collision: bool = False, backward_collision: bool = False, exclude_path: bool = True, **kwargs):
        new = []
        if exclude_path:
            new = self.world.get_neighbours(self.state, exclude=self.path)
        else:
            new = self.world.get_neighbours(self.state)
        if collision_check:
            collidable_states = None
            if own_colony_collision:
                collidable_states = [x.state for x in world.agents if x is not self and (backward_collision and self.forward)]
            else:
                collidable_states = [x.state for x in world.agents if x.colony is not self.colony]
            new = [x for x in new if x not in collidable_states]
        return new

    def transition_probabilities(self, new, state=None, **kwargs):
        if state is None:
            state = self.state
        return {k: self.transition_value(state, k, **kwargs) for k in new}


    def decision(self, **kwargs) -> int:
        """
        :param kwargs:
        :type alpha: float
        :type beta: float
        :return: decision for next state
        """
        new = self.transition_options(**kwargs)
        if len(new) == 0:
            self.stuck = True
            return None
        return fitness_proportional_selection(self.transition_probabilities(new, **kwargs), random=self.random.rand())


    def _put_pheromones_to_nodes(self, path, amount):
        for i, j in zip(path[:-1], path[1:]):
            #deposit amount for a specific edge
            self.colony.pheromones[i, j] += amount / len(path)

    def put_pheromones(self, c_t=1.0, c_d=1.0, **_):
        path = self.path
        if not self.forward:
            path.reverse()

        assert path[0] == self.start
        assert path[-1] == self.goal
        #amount of pheromones to deposit on the whole path
        amount = c_t / len(path) + c_d / self.world.path_distance(path)
        self._put_pheromones_to_nodes(path, amount)

    def delayed_pheromone_update(self, **kwargs) -> None:
        if self.arrived:
            self.put_pheromones(**kwargs)

    def pheromone_update(self, elitism_amount=0.0, **kwargs):
        self.delayed_pheromone_update(**kwargs)
        if elitism_amount != 0.0 and self.best_path is not None:
            self._put_pheromones_to_nodes(self.best_path, elitism_amount)
        self.colony.vaporize(**kwargs)
        self.colony.dissipate(**kwargs)

    @property
    def graph_str(self) -> str:
        s = self.name
        if not self.forward:
            s = s + "*"
        return s

    @property
    def arrived(self) -> bool:
        if self.forward and self.state == self.goal:
            return True
        elif not self.forward and self.state == self.start:
            return True
        return False

    def reset(self, reverse_ant_on_reset=True, **_):
        if reverse_ant_on_reset and not self.stuck:
            self.forward = not self.forward
        self.state = self.start if self.forward else self.goal
        self.path = [self.state]
        self.stuck = False

    def check_best_path(self, path):
        if self.world.path_distance(path) < self.world.path_distance(self.colony.best_path):
            self.colony.best_path = path
        if self.world.path_distance(path) < self.world.path_distance(self.best_path):
            self.best_path = path
        if self.world.path_distance(path) < self.world.path_distance(self.world.best_path):
            self.world.best_path = path


    def daemon_actions(self, **kwargs):
        if self.arrived:
            self.arrived_counter += 1
            path = self.path
            if not self.forward:
                path.reverse()
            self.check_best_path(path)
        if self.arrived or self.stuck:
            if self.stuck:
                self.stuck_counter += 1
            self.reset(**kwargs)
        if self.greedy_path_dist < self.world.path_distance(self.best_path):
            self.check_best_path(self.greedy_path)

    def step(self, **kwargs):
        self.step_count += 1
        self.kwargs.update(kwargs)
        self.state = self.decision(**self.kwargs)
        self.path.append(self.state)
        self.pheromone_update(**self.kwargs)
        self.daemon_actions(**self.kwargs)

    @property
    def best_time(self):
        if self.best_path is None:
            return np.nan
        return len(self.best_path)

    @property
    def greedy_path(self):
        # caching the path for each step
        if self._greedy_path_step == self.step_count:
            return self._greedy_path
        path = [self.start]
        while path[-1] != self.goal:
            new = self.world.get_neighbours(path[-1], exclude=path)
            if len(new) < 1:
                self._greedy_path_step = self.step_count
                self._greedy_path = path
                return path
            probs = self.transition_probabilities(new, state=path[-1], forward=True, **self.kwargs)
            best_value = 0
            best_key = 0
            for k, v in probs.items():
                if v > best_value:
                    best_value = v
                    best_key = k
            path.append(best_key)
        # updating the cache
        self._greedy_path_step = self.step_count
        self._greedy_path = path
        return path

    @property
    def greedy_path_dist(self):
        p = self.greedy_path
        #print(f"{p}, {self.goal}")
        if p[-1] == self.goal:
            return self.world.path_distance(p)
        return np.nan

    @property
    def greedy_path_time(self):
        p = self.greedy_path
        if p[-1] == self.goal:
            return len(p)
        return np.nan


class TimeAwareAnt(AcoAgent):
    def __init__(self, time_frame: int = 50, **kwargs):
        AcoAgent.__init__(self, **kwargs)
        self.time_frame = time_frame

    def initialize_pheromones(self):
        AcoAgent.initialize_pheromones(self)
        self.colony.time_pheromones = np.zeros((self.time_frame, self.colony.pheromones.shape[0], self.colony.pheromones.shape[1]))
        for t in range(self.time_frame):
            self.colony.time_pheromones[t] = self.colony.pheromones.copy()


    @property
    def time(self):
        return len(self.path)

    def time_transition_value(self, i, j, time: int = None, forward: bool = None, alpha_time: float = 1.0, alpha: float = 1.0, beta: float = 1.0, eps: float = 0.01, **_):
        if time is None:
            time = self.time
        return self.colony.time_pheromones[time, i, j] ** alpha_time + self.transition_value(i, j, forward=forward, alpha=alpha, beta=beta, eps=eps)

    def time_transition_probabilities(self, new, state=None, **kwargs):
        if self.forward:
            return self.transition_probabilities(new, state=state, **kwargs)
        if state is None:
            state = self.state
        return {self.time_transition_value(state, s, **kwargs) for s in new}

    def _put_pheromones_to_nodes(self, path, amount):
        AcoAgent._put_pheromones_to_nodes(self, path, amount)
        if not self.forward or len(path) > self.time_frame:
            return
        delta = amount / len(path)
        # todo: efficiency
        while len(path) < self.time_frame - 1:
            path.append(self.goal)
        #print(f"{len(path)} : {path}")

        #print(path)
        for i, to in enumerate(path[1:]):
            #print(f"{amount}, {i}, {to}, {path[i]}")
            self.colony.time_pheromones[i, path[i], to] += delta

    def decision(self, time_aware=True, **kwargs) -> int:
        if not self.forward or not time_aware:
            return AcoAgent.decision(self, **kwargs)
        if not self.time < self.time_frame - 1:
            self.stuck = True
            return None
        new = self.transition_options(exclude_path=False, **kwargs)
        probs = self.time_transition_probabilities(new, **kwargs)
        return fitness_proportional_selection(probs, self.random.rand())


if __name__ == '__main__':
    world = TestProblem().easy_1()
    colony1 = TimeAwareColony()
    colony2 = Colony()
    agents = [TimeAwareAnt(colony=colony1, start=world.agents[0].start, goal=world.agents[0].goal,
                           time_frame=12, dissipation_rate=0.01, time_dissipation_rate=0.0001, elitism_amount=0.0, evaporation_method="normalize")
              for _ in range(2)]
    #agents += [AcoAgent(colony=colony2, start=10, goal=20) for _ in range(2)]
    #agents += [AcoAgent(colony=colony2, start=20, goal=30) for _ in range(2)]
    world.update_agents(agents)
    for _ in range(100):
        world.step(c_t = 0.1, c_d = 0.1, alpha = 1.0, beta=1.0, time_aware=False)
    print(world.get_data())
    for _ in range(2000):
        world.step(c_t = 0.1, c_d = 0.3, alpha = 1.0, beta=0, alpha_time = 0.5, time_aware=True)
    #print(f"{colony.pheromones}")
    print(world.get_data())
    world.dot_graph(colony1.pheromones , render=True, normalize_pheromone=True)
    world.dot_graph(colony1.time_pheromones[1], render=True, normalize_pheromone=True, eps=0.0)
    world.dot_graph(colony1.time_pheromones[2], render=True, normalize_pheromone=True, eps=0.0)
    world.dot_graph(colony1.time_pheromones[3], render=True, normalize_pheromone=True, eps=0.0)
