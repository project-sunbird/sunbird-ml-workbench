Requirements
---
python 2.7 and pip
virtualenv - https://virtualenv.pypa.io/en/stable/installation/

How to install
---
1. Create virtual environment - 'virtualenv run_env'
2. Activate virtual environment - 'source run_env/bin/activate'
3. Install - 'pip install  <install>/mlworkbench-0.0.1.tar.gz'

How to run a dag - Iris classification example
---
1. Activate virtual environment in which mlworkbench is installed - 'source run_env/bin/activate' - skip if already in the virtual environment
2. Optionally declare a working directory for MLWB  - 'export MLWB_HOME=<directory location>' (default: ~/MLWB_HOME)
3. 'mlworkbench run -dag <location of iris_experiment.yaml>' (examples/Iris_Classification/iris_experiment.yaml)

+ localhost:8080 has airflow webserver visualization of the DAG
+ 'Ctrl + C' will kill the mlworkbench run process (This will work only after the DAG has been submitted)  

Note: Graph inputs, outputs and experiment directory locations are defined relative to the yaml file location


How to build and install
---
1. Run 'bash build.sh' in 'mlworkbench' folder
2. Create virtual environment - 'virtualenv run_env'
3. Activate virtual environment - 'source run_env/bin/activate' 
4. Install - 'pip install  <mlworkbench>/target/dist/mlworkbench-0.0.1/dist/mlworkbench-0.0.1.tar.gz'

