from src.aco_mapf.GraphWorld import *

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
    normalized = m / np.sum(m)

class AcoAgent(NavigationAgent):
    pheromones: np.matrix
    data: list
    random: np.random.RandomState

    def __init__(self, seed=None, **kwargs):
        NavigationAgent.__init__(self, **kwargs)
        self.data = []
        self.path = [self.state]
        self.random = np.random.RandomState(seed=seed)
        self.forward = True

    def initialize_pheromones(self):
        self.pheromones = np.ones_like(self.world.adjacency)


    def transition_value(self, i, j, forward=True, alpha: float = 1.0, **kwargs):
        """

        :type alpha: float
        """
        if not forward:
            i, j = j, i
        return self.pheromones[i, j]**alpha

    def decision(self, **kwargs) -> int:
        """

        :param kwargs:
        :type alpha: float
        :return: decision for next state
        """
        new = self.world.get_neighbours(self.state)
        new = [next for next in new if next not in self.path]
        probs = {k: self.value(self.state, k, **kwargs) for k in new}
        return fitness_proportional_selection(probs, random=self.random.rand())

    def delayed_pheromone_update(self):
        pass

    def vaporize(self):
        self.pheromones = normalize_matrix(self.pheromones)

    def pheromone_update(self):
        self.delayed_phermone_update()
        self.vaporize()

    def daemon_actions(self):
        pass

    def step(self):
        self.state = self.decision()
        self.pheromone_update()
        self.daemon_actions()

