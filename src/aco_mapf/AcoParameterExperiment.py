from ExperimentRunner.ExperimentRunner import *
import sys
import ipyparallel as ipp
from src.aco_mapf.AcoAgent import AcoAgent, Colony
from src.aco_mapf.GraphWorld import TestProblem

@ipp.require("pandas as pd", "numpy as np", TestProblem, AcoAgent, Colony)
def run_testprolem_aco(seed=0, num_agents=1,**kwargs):
    c = Colony()
    agents = [AcoAgent(seed=seed+offset, colony=c, **kwargs) for offset in range(num_agents)]
    problem = TestProblem(seed=seed).hard_1(agents=agents)
    data = []
    for _ in range(10):
        for _ in range(50):
            problem.step()
        data.append(problem.get_data())
    return pd.concat(data)


def run_experiment(name, interactive=False):
    print(f"running experiment for parameters {name}.json, saving to {name}.pkl")
    experiment = Experiment(function=run_testprolem_aco)
    experiment.load_parameters(f"{name}.json")
    print(f"Got {len(experiment.parameters)} parameters.")
    experiment.generate_tasks()
    print(f"Generating running {len(experiment.tasks)} tasks.")
    experiment.run_map(interactive=True)
    print("saving")
    experiment.save_results(f"{name}.pkl")

if __name__ == "__main__":
    run_experiment(sys.argv[-1])