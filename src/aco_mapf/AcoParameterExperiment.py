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
    df = pd.DataFrame(data)
    return df.replace(np.inf, np.nan)


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

def run_experiment(filename, runs=31, data_file=None, **kwargs):
    experiment = Experiment(function=run_testprolem_aco,
                            param_file=filename,
                            runs=runs, **kwargs)
    experiment.generate_tasks()
    experiment.run_map()
    if data_file:
        experiment.save_results(data_file)

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
            print("\n\n")
        if data_file:
            optimizer.save_results(data_file)
        if outfile:
            optimizer.save_parameters(outfile)
    except KeyboardInterrupt:
        print("interrupted by keyboard")
    print(f"best run was: {optimizer.global_best_identifier}")
    return optimizer


if __name__ == "__main__":
    run_optimization("optimize.json", runs=1, with_cluster=False)
