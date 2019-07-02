RUNS=3
GENERATIONS=15
POPULATION=10
PROPERTY=greedy_distance
STEP=1000

all: *_optimize.pkl


%_optimize.pkl: %_optimize.json
	../run_experiments.py --optimize $< --out $@ --runs 3 --generations 15 --population 10 --property greedy_distance --median --step 1000
