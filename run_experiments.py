#!bin/python3

import argparse
from ExperimentRunner.ExperimentRunner import *
from src.aco_mapf.AcoParameterExperiment import *

def optimize(args):
    outfile = args.optimize
    datafile = None
    eval_kwargs = {}
    if args.data_out:
        datafile = args.data_out
    if args.params_out:
        outfile = args.params_out
    if args.average:
        eval_kwargs["average"] = args.average
    if args.property:
        eval_kwargs["property"] = args.property
    if args.step:
        eval_kwargs["step"] = args.step
    run_optimization(args.optimize, runs=args.runs, generations=args.generations, outfile=outfile, data_file=datafile, with_cluster=args.parallel, population_size=args.population)

def analyze(args):
    datafile = None
    if args.data_out:
        datafile = args.data_out
    run_optimization(args.analyze, runs=args.runs, data_file=datafile, with_cluster=args.parallel)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", help='file of parameters to optimize', type=str)
    parser.add_argument("--data_out", help='file name for data output')
    parser.add_argument("--params_out", help='file name to write back parameters, default is to write back to the input!')
    parser.add_argument("--runs", help="number of runs", default=31, type=int)
    parser.add_argument("--generations", help="number of generations to run", default=20, type=int)
    parser.add_argument("--population", help="population size", default=10, type=int)
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--sequential", action="store_false", dest="parallel")
    parser.add_argument("--step", type=int, help="step number of the fitness evaluation")
    parser.add_argument("--average", help="used as the averaging method: either mean or median")
    parser.add_argument("--median", help="use median as averaging method in evaluation", action="store_const", const="median", dest="average")
    parser.add_argument("--property", help="property to use in the evaluation i.e. min_best_distance")
    parser.add_argument("--analyze", help='file of parameters to analyze', type=str)
    args = parser.parse_args()
    if args.optimize:
        optimize(args)

    if args.analyze:
        analyze(args)