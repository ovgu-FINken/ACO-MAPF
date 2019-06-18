#!bin/python3

import argparse
from ExperimentRunner.ExperimentRunner import *
from src.aco_mapf.AcoParameterExperiment import *

def optimize(args):
    outfile = args.optimize
    datafile = None
    if args.data_out:
        datafile = args.out
    if args.params_out:
        outfile = args.params_out
    run_optimization(args.optimize, runs=args.runs, generations=args.generations, outfile=outfile, data_file=datafile, with_cluster=args.parallel)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", required=True, help='file of parameters to optimize')
    parser.add_argument("--data_out", help='file name for data output')
    parser.add_argument("--params_out", help='file name to write back parameters, default is to write back to the input!')
    parser.add_argument("--runs", help="number of runs", default=31)
    parser.add_argument("--generations", help="number of generations to run", default=20)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--sequential", action="store_false", dest="parallel")
    args = parser.parse_args()
    if args.optimize:
        optimize(args)
