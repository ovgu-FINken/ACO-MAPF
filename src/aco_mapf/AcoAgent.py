from src.aco_mapf.GraphWorld import *
from src.aco_mapf.GraphWorld import NavigationAgent


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

    def transition_options(self):
        return self.world.get_neighbours(self.state, exclude=self.path)

    def transition_probabilities(self, new, **kwargs):
        return {k: self.transition_value(self.state, k, **kwargs) for k in new}


    def decision(self, **kwargs) -> int:
        """

        :param kwargs:
        :type alpha: float
        :type beta: float
        :return: decision for next state
        """
        new = self.transition_options()
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

    def vaporize(self):
        self.colony.pheromones = normalize_matrix(self.colony.pheromones)

    def pheromone_update(self, elitism_amount=0.0,**kwargs):
        self.delayed_pheromone_update(**kwargs)
        if elitism_amount != 0.0 and self.best_path is not None:
            self._put_pheromones_to_nodes(self.best_path, elitism_amount)
        self.vaporize()

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

    def daemon_actions(self, **kwargs):
        if self.arrived:
            self.arrived_counter += 1
            path = self.path
            if not self.forward:
                path.reverse()
            if self.world.path_distance(path) < self.world.path_distance(self.colony.best_path):
                self.colony.best_path = path
            if self.world.path_distance(path) < self.world.path_distance(self.best_path):
                self.best_path = path
            if self.world.path_distance(path) < self.world.path_distance(self.world.best_path):
                self.world.best_path = path
        if self.arrived or self.stuck:
            if self.stuck:
                self.stuck_counter += 1
            self.reset(**kwargs)

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


if __name__ == '__main__':
    world = TestProblem().hard_2()
    colony1 = Colony()
    colony2 = Colony()
    agents = [AcoAgent(colony=colony1, start=world.agents[0].start, goal=world.agents[0].goal, elitism_amount=0.1) for _ in range(2)]
    agents += [AcoAgent(colony=colony2, start=10, goal=20) for _ in range(2)]
    agents += [AcoAgent(colony=colony2, start=20, goal=30) for _ in range(2)]
    world.update_agents(agents)
    for _ in range(1000):
        world.step(c_t = 0.1, c_d = 0.1)
    #print(f"{colony.pheromones}")
    print(world.get_data())
    world.dot_graph(colony1.pheromones , render=True)