import networkx as nx
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import graphviz
import tempfile


class NavigationAgent(object):

    def __init__(self, start: int = 0, goal: int = 1, state: int = None):
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

    def draw_pheromones(self,
                        pheromone_matrix: np.matrix,
                        cutoff: float = 0.01,
                        thickness: float = 1.0,
                        node_size: float = 1.0,
                        node_label: bool = False,
                        label: bool = True
                        ):
        matrix = pheromone_matrix.copy()
        for i, j in np.ndindex(matrix.shape):
            if matrix[i, j] < cutoff:
                matrix[i, j] = 0
        G = nx.from_numpy_matrix(matrix, create_using=nx.DiGraph())
        weight = nx.get_edge_attributes(G, 'weight')
        node_colors = ["lightblue" for _ in range(matrix.shape[0])]
        labels = {k: f"{v:.2f}" for k, v in weight.items()}
        if node_label:
            node_labels = None
            if label:
                node_labels = {k: f"{np.diag(matrix)[k]:.2f}" for k in range(matrix.shape[0])}
            nx.draw_networkx_labels(G, pos=self.graph_pos, labels=node_labels)

        if label:
            nx.draw_networkx_edge_labels(G, self.graph_pos, edge_labels=labels, label_pos=0.3)
        width = [(w['weight']) * thickness / np.max(matrix) for u, v, w in G.edges(data=True)]
        nx.draw_networkx_nodes(G, pos=self.graph_pos, node_color=node_colors,
                               node_size=[node_size * 300 * (1 + v) for v in np.diag(matrix)])
        nx.draw_networkx_edges(G, self.graph_pos, width=width, alpha=0.3, connectionstyle='arc3,rad=0.2', arrowstyle="-|>",
                               arraowsize=thickness * 3)

    def get_neighbours(self, state: int, exclude: List[int] = []) -> List[int]:
        if state is None:
            raise ValueError
        neighbours = []
        for i in range(self.nodes):
            if self.adjacency[state, i] > 0 and i not in exclude:
                neighbours.append(i)
        return neighbours

    def dot_graph(self, pheromones: np.matrix = None, eps: float = 0.01, render=False):
        dot = graphviz.Digraph(comment="representation of current world state", engine="neato")
        dot.attr(overlap="scale")
        dot.attr(K="1")
        dot.attr(maxiter="20000")
        dot.attr(start="42")
        for i in range(self.adjacency.shape[0]):
            dot.node(f"{i}", f"{i}")
        for (i, j), v in np.ndenumerate(self.adjacency):
            if v > 0:
                label = f"{v}"
                color="black"
                if pheromones is not None:
                    color = f"gray{int(90 - 90 * pheromones[i,j] / np.max(pheromones))}"
                    if pheromones[i, j] <= eps:
                        label = ""
                    else:
                        label = f"{pheromones[i,j]:.2f}"
                dot.edge(f"{i}", f"{j}", label=label, color=color)

        for i in range(self.adjacency.shape[0]):
            state = []
            start = []
            goal = []
            for j, agent in enumerate(self.agents):
                if agent.state == i:
                    state.append(j)
                if agent.start == i:
                    start.append(j)
                if agent.goal == i:
                    goal.append(j)
            if len(state) > 0:
                s = f"state: {state}"
                dot.node(f"state{i}", label=s, color="blue")
                dot.edge(f"state{i}", f"{i}", color="blue")
            if len(start) > 0:
                s = f"start: {start}"
                dot.node(f"start{i}", label=s, color="green")
                dot.edge(f"start{i}", f"{i}", color="green")
            if len(goal) > 0:
                s = f"goal: {goal}"
                dot.node(f"goal{i}", label=s, color="red")
                dot.edge(f"goal{i}", f"{i}", color="red")
        if render:
            dot.render(tempfile.mktemp('.gv'), view=True)
        return dot

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

    def graph_prolem(self, G, start=0, goal=1, agents=None, **_) -> GraphWorld:
        if agents is None:
            agents = [NavigationAgent(start=start, goal=goal)]
        else:
            for agent in agents:
                agent.start = start
                agent.goal = goal
                agent.state = start
        return GraphWorld(adjacency=nx.adj_matrix(G).todense(), agents=agents)

    def watts_strogatz_problem(self, nodes, k, p, seed=42, start=0, goal=1, **kwargs):
        G = nx.watts_strogatz_graph(nodes, k, p, seed=seed)
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
    from src.aco_mapf.AcoAgent import *
    colony = Colony()
    agents = [AcoAgent(seed = i, colony=colony) for i in range(10)]
    world = TestProblem().hard_1(agents=agents)
    for _ in range(300):
        world.step()
