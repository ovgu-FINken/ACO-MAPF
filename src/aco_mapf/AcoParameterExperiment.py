from ExperimentRunner.ExperimentRunner import *
import sys

def run_function():
    return []

if __name__ == "__main__":
    name = sys.argv[-1]
    print(f"running experiment for parameters {name}.json, saving to {name}.pkl")
    experiment = Experiment(function=run_function)
    experiment.load_parameters(f"{name}.json")
    #experiment.run_map()
    print("running")
    print("saving")
    experiment.save_results(f"{name}.pkl")
