RUNS=3
GENERATIONS=15
POPULATION=10
PROPERTY=greedy_distance
STEP=1000
VENV_NAME?=..
VENV_ACTIVATE=$(VENV_NAME)/bin/activate
PYTHON=${VENV_NAME}/bin/python3


all: *_optimize.pkl

venv: $(VENV_NAME)/bin/activate
$(VENV_NAME)/bin/activate: #setup.py
	test -d $(VENV_NAME) || virtualenv -p python3 $(VENV_NAME)
	${PYTHON} -m pip install -U pip
	${PYTHON} -m pip install -e .
	${PYTHON} -m pip install --upgrade git+https://github.com/ovgu-FINken/python-experiment-runner.git
	touch $(VENV_NAME)/bin/activate

%_optimize.pkl: %_optimize.json venv
	${PYTHON} ../run_experiments.py --optimize $< --out $@ --runs 3 --generations 15 --population 10 --property greedy_distance --median --step 1000
