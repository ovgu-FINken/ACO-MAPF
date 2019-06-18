from ExperimentRunner.ExperimentRunner import *
import sys
import ipyparallel as ipp
from src.aco_mapf.AcoAgent import AcoAgent, Colony
from src.aco_mapf.GraphWorld import TestProblem

@ipp.require("pandas as pd", "numpy as np", TestProblem, AcoAgent, Colony)
def run_testprolem_aco(seed=0, num_agents=1, log_steps=20, between_log_steps=50,**kwargs):
    c = Colony()
    agents = [AcoAgent(seed=seed+offset, colony=c, **kwargs) for offset in range(num_agents)]
    problem = TestProblem(seed=seed).hard_1(agents=agents)
    data = []
    for _ in range(log_steps):
        for _ in range(between_log_steps):
            problem.step()
        data.append(problem.get_data())
    return pd.concat(data)


def eval_testproblem_aco(df):
    print(df.keys())
    fitness = df["min_best_distance"].mean()
    return fitness


def run_experiment(name, interactive=False):
    print(f"running experiment for parameters {name}.json, saving to {name}.pkl")
    experiment = Experiment(function=run_testprolem_aco, param_file=f"{name}.json")
    print(f"Got {len(experiment.parameters)} parameters.")
    experiment.generate_tasks()
    print(f"Generating running {len(experiment.tasks)} tasks.")
    experiment.run_map(interactive=True)
    print("saving")
    experiment.save_results(f"{name}.pkl")

def run_optimization(name, generations=20, runs=31, **kwargs):
    optimizer = Optimizer(function=run_testprolem_aco, evaluation_function=eval_testproblem_aco, population_size=10, param_file=f"{name}.json", runs=runs, **kwargs)
    optimizer.init_population(10)
    try:
        for _ in range(generations):
            print(f"{optimizer.generation} optimzer.best: {optimizer.global_best_fitness}\n{optimizer.global_best}")
            print([str(p) for p in optimizer.mapping])
            print("\n\n\n")
            optimizer.run_generation()
        optimizer.save_results(f"{name}.pkl")
        optimizer.save_parameters(f"{name}.json")
    except KeyboardInterrupt:
        print("interrupted ...")
    return optimizer


if __name__ == "__main__":
    run_optimization("optimize", runs=1, with_cluster=False)
