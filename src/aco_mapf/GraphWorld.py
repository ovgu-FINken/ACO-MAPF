import networkx as nx
import numpy as np
from typing import List

class NavigationAgent:

    def __init__(self, start: int = 0, goal: int = 1, state: int = None) -> object:
        self.start = start
        self.goal = goal
        if state is None:
            self.state = start
        else:
            self.state = state

    def register_world(self, world):
        """
        register the world to the agent
        :type world: GraphWorld
        """
        assert isinstance(world, GraphWorld)
        self.world = world


class GraphWorld:
    adjacency: np.matrix

    def __init__(self, adjacency=None, agents=None):
        self.adjacency = adjacency
        self.graph = nx.from_numpy_matrix(self.adjacency)
        self.graph_pos = nx.kamada_kawai_layout(self.graph)
        self.agents = agents
        if self.agents is None:
            self.agents = [NavigationAgent()]

        for agent in self.agents:
            agent.register_world(self)

    def draw_adjacency(self):
        return nx.draw(self.graph, self.graph_pos)

    def get_neighbours(self, state:int) -> List[int]:
        neighbours = []
        for i in range(self.nodes):
            if self.adjacency[state, i] > 0:
                neighbours.append(i)
        return neighbours


    @property
    def nodes(self):
        return self.adjacency.shape[0]

    @property
    def egdes(self):
        return len(self.graph.edges)


class TestProblem:
    def __init__(self, seed=None):
        self.random = np.random.RandomState(seed=seed)

    def graph_prolem(self, G, start=0, goal=1, **kwargs):
        return GraphWorld(adjacency=nx.adj_matrix(G).todense(), agents=[NavigationAgent(start=start, goal=goal)])

    def watts_strogatz_problem(self, nodes, k, p, start=0, goal=1, **kwargs):
        G = nx.watts_strogatz_graph(nodes, k, p, **kwargs)
        for e in G.edges():
            G[e[0]][e[1]]['weight']= 0.5 + self.random.rand()

        return self.graph_prolem(G, start=start, goal=goal, **kwargs)

    def hard_1(self):
        return self.watts_strogatz_problem(145,4, 0.01, seed=42, start=0, goal=75)
        G = nx.watts_strogatz_graph(145,4, 0.01, seed=42)

    def hard_2(self):
        return self.watts_strogatz_problem(42,4,0.05, seed=23, goal=27)

    def easy_1(self):
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
        self.graph_prolem(G, goal=8)

#G.add_edge(5, 9)

    def easy_2(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=1.5)
        G.add_edge(1, 4, weight=1.0)
        return self.graph_prolem(G, goal=3)

    def easy_3(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(0, 2, weight=1.0)
        G.add_edge(1, 3, weight=1.0)
        G.add_edge(2, 3, weight=1.0)
        return self.graph_prolem(G, goal=3)

    def easd_4(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(2, 3, weight=1.0)
        G.add_edge(2, 4, weight=1.0)
        G.add_edge(4, 5, weight=1.0)
        G.add_edge(5, 7, weight=1.0)
        G.add_edge(7, 6, weight=1.0)
        G.add_edge(4, 6, weight=1.0)
        G.add_edge(6, 8, weight=1.0)
        return self.graph_prolem(G, goal=7)

if __name__ == "__main__":
    world = TestProblem().hard_1()
    print(f"nodes: {world.nodes}, edges: {world.egdes}")


