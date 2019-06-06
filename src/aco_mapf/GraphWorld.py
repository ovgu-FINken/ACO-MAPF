import networkx as nx
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import pandas as pd

class NavigationAgent:

    def __init__(self, start: int = 0, goal: int = 1, state: int = None, id=0) -> object:
        self.start = start
        self.goal = goal
        if state is None:
            self.state = start
        else:
            self.state = state

    def step(self):
        pass

    def register_world(self, world):
        """
        register the world to the agent
        :type world: GraphWorld
        """
        assert isinstance(world, GraphWorld)
        self.world = world


class GraphWorld:
    adjacency: np.matrix
    best_path: List[int]

    def __init__(self, adjacency=None, agents=None):
        self.adjacency = adjacency
        self.graph = nx.from_numpy_matrix(self.adjacency)
        self.graph_pos = nx.kamada_kawai_layout(self.graph)
        self.agents = agents
        self.best_path = None
        if self.agents is None:
            self.agents = [NavigationAgent()]

        for agent in self.agents:
            agent.register_world(self)

    def update_agents(self, agents: NavigationAgent):
        """

        :type agents: NavigationAgent
        """
        self.agents = agents
        for agent in self.agents:
            agent.register_world(self)

    def draw_adjacency(self):
        return nx.draw(self.graph, self.graph_pos, labels={k : str(k) for k in self.graph.nodes()} )


    def get_neighbours(self, state: int, exclude: List[int] = []) -> List[int]:
        if state is None:
            raise ValueError
        neighbours = []
        for i in range(self.nodes):
            if self.adjacency[state, i] > 0 and i not in exclude:
                neighbours.append(i)
        return neighbours

    @property
    def nodes(self):
        return self.adjacency.shape[0]

    @property
    def egdes(self):
        return len(self.graph.edges)

    def step(self):
        for agent in self.agents:
            agent.step()

    def path_distance(self, path: List[int]) -> float:
        d: float = 0.0
        if path is None:
            return np.inf
        for i, j in zip(path[:-1], path[1:]):
            d += self.adjacency[i, j]
        return d

    @property
    def max_best_distance(self):
        return self.path_distance(self.best_path)

    @property
    def median_best_distance(self):
        return np.median([self.path_distance(a.best_path) for a in self.agents])

    def get_data(self):
        return pd.DataFrame([{
            "median_best_distance": self.median_best_distance,
            "max_best_distance" : self.max_best_distance
        }])

class TestProblem:
    def __init__(self, seed=None):
        self.random = np.random.RandomState(seed=seed)

    def graph_prolem(self, G, start=0, goal=1, agents=None,**_):
        if agents is None:
            agents = [NavigationAgent(start=start, goal=goal)]
        else:
            for agent in agents:
                agent.start = start
                agent.goal = goal
                agent.state = start
        return GraphWorld(adjacency=nx.adj_matrix(G).todense(), agents=agents)

    def watts_strogatz_problem(self, nodes, k, p, start=0, goal=1, **kwargs):
        G = nx.watts_strogatz_graph(nodes, k, p)
        for e in G.edges():
            G[e[0]][e[1]]['weight'] = 0.5 + self.random.rand()

        return self.graph_prolem(G, start=start, goal=goal, **kwargs)

    def hard_1(self, **kwargs):
        return self.watts_strogatz_problem(145, 4, 0.01, seed=42, start=0, goal=75, **kwargs)

    def hard_2(self, **kwargs):
        return self.watts_strogatz_problem(42, 4, 0.05, seed=23, goal=27, **kwargs)

    def easy_1(self, **kwargs):
        G = nx.Graph()
        G.add_edge(0, 1, weight=1)
        G.add_edge(1, 2, weight=1.5)
        G.add_edge(1, 3, weight=1)
        G.add_edge(3, 5, weight=1)
        G.add_edge(1, 4, weight=3)
        G.add_edge(2, 5, weight=1)
        G.add_edge(4, 6, weight=1)
        G.add_edge(5, 7, weight=1)
        G.add_edge(6, 7, weight=3)
        G.add_edge(7, 8, weight=1)
        return self.graph_prolem(G, goal=8, **kwargs)

    # G.add_edge(5, 9)

    def easy_2(self, **kwargs):
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=1.5)
        G.add_edge(1, 4, weight=1.0)
        return self.graph_prolem(G, goal=3, **kwargs)

    def easy_3(self, **kwargs):
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(0, 2, weight=1.0)
        G.add_edge(1, 3, weight=1.0)
        G.add_edge(2, 3, weight=1.0)
        return self.graph_prolem(G, goal=3, **kwargs)

    def easy_4(self, **kwargs):
        G = nx.Graph()
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(2, 3, weight=1.0)
        G.add_edge(2, 4, weight=1.0)
        G.add_edge(4, 5, weight=1.0)
        G.add_edge(5, 7, weight=1.0)
        G.add_edge(7, 6, weight=1.0)
        G.add_edge(4, 6, weight=1.0)
        G.add_edge(6, 8, weight=1.0)
        return self.graph_prolem(G, goal=7, **kwargs)


if __name__ == "__main__":
    world = TestProblem().hard_1()
    print(f"nodes: {world.nodes}, edges: {world.egdes}")
    plt.figure()
    world.draw_adjacency()
    plt.show()
