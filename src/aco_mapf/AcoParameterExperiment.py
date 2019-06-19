from ExperimentRunner.ExperimentRunner import *
import ipyparallel as ipp
from src.aco_mapf.AcoAgent import AcoAgent, Colony
from src.aco_mapf.GraphWorld import TestProblem
import functools

@ipp.require("pandas as pd", "numpy as np", TestProblem, AcoAgent, Colony)
def run_testprolem_aco(seed=0, num_agents=1, log_steps=20, between_log_steps=50, problem="hard_1", **kwargs):
    c = Colony()
    agents = [AcoAgent(seed=seed+offset, colony=c, **kwargs) for offset in range(num_agents)]
    #problem = None
    if problem == "hard_1":
        problem = TestProblem(seed=seed).hard_1(agents=agents)
    elif problem == "hard_2":
        problem = TestProblem(seed=seed).hard_2(agents=agents)
    elif problem == "easy_1":
        problem = TestProblem(seed=seed).easy_1(agents=agents)
    elif problem == "easy_2":
        problem = TestProblem(seed=seed).easy_2(agents=agents)
    elif problem == "easy_3":
        problem = TestProblem(seed=seed).easy_3(agents=agents)
    elif problem == "easy_4":
        problem = TestProblem(seed=seed).easy_4(agents=agents)
    data = []
    for _ in range(log_steps):
        for _ in range(between_log_steps):
            problem.step()
        data.append(problem.get_data())
    return pd.concat(data)


def eval_min_best_distance_mean(df):
    #print(df.keys())
    fitness = df["min_best_distance"].mean()
    return fitness


def eval_df(data, step=None, avg="mean", property="min_best_distance", verbose=False, **kwargs):
    if(len(kwargs) > 0):
        print(f"eval got unrecognized kwargs:\n{kwargs}")
    df = data
    if step is not None:
        if step > 0:
            df = df.loc[df.world_steps == step]
        elif step < 0:
            raise NotImplementedError
    fitness = np.nan
    if avg == "median":
        fitness = df[property].median()
    fitness = df[property].mean()
    if verbose:
        print(f"fitness: {fitness}")
    return fitness

def run_experiment(name):
    print(f"running experiment for parameters {name}.json, saving to {name}.pkl")
    experiment = Experiment(function=run_testprolem_aco, param_file=f"{name}.json")
    print(f"Got {len(experiment.parameters)} parameters.")
    experiment.generate_tasks()
    print(f"Generating running {len(experiment.tasks)} tasks.")
    experiment.run_map()
    print("saving")
    experiment.save_results(f"{name}.pkl")

def run_optimization(filename, generations=20, runs=31, outfile=None, data_file=None, eval_kwargs={}, population_size=10, **kwargs):
    optimizer = Optimizer(function=run_testprolem_aco,
                          evaluation_function=functools.partial(eval_df, **eval_kwargs),
                          population_size=population_size,
                          param_file=filename,
                          runs=runs, **kwargs)
    try:
        for _ in range(generations):
            print(f"{optimizer.generation} optimizer.best: {optimizer.global_best_fitness}\n{optimizer.global_best}")
            print("\n".join([f"{p.name} -- {p.best} from [{p.low}, {p.high}]" for p in optimizer.mapping]))
            print(optimizer.population)
            optimizer.run_generation()
            print(optimizer.fitness)
            print("\n\n\n")
        if data_file:
            optimizer.save_results(data_file)
        if outfile:
            optimizer.save_parameters(outfile)
    except KeyboardInterrupt:
        print("interrupted ...")
    return optimizer


if __name__ == "__main__":
    run_optimization("optimize.json", runs=1, with_cluster=False)
