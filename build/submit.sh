# options
# -validate: will validate the dag

# Usage

# Default options

dag_file='./dag.yaml'
dag_folder=`/tmp/dag_files`
working_dir='./working_dir'

if[!$1]
    echo "./submit.sh [-f dag_file.yaml] [-folder <dag_folder>] [-ids <comma separated dag ids>] [-wd <working-dir>] [-dgf <dag folder>] [--dryrun]";
    exit 1;
end if

# Two options

## Option 1

### 1. copy the yaml file to config.dag_files path (default to /tmp/dag_files)
### 2. "airflow initdb" to store the parsed dags
### 3. <dag-id> = Read the yaml file and get the dag id
### 4. airflow unpause <dag-id>

### cd examples/sample
### mlworkbench/submit.sh -f example/sample/dag.yaml -ids 'Iris_classifier,Sample_classifier'

## Option 2
## 1. creat a working dir under the basedir /workdir 
## 2. Parse the yaml file and put it under the /workdir
## 3. "airflow run -sd <work-dir> <dag-id>"