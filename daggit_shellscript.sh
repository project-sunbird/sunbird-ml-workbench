#!/bin/bash

# cd /home/aleena/data
# echo "Hello world"
# source daggit_testrun/bin/activate

#cd /home/aleena/Desktop/ML-Workbench
daggit init dag_examples/task_distribution/task_distribution_1.yaml 
daggit run TaskDistribution_experiment_1 
airflow clear TaskDistribution_experiment_1
daggit init dag_examples/task_distribution/task_distribution_2.yaml
daggit run TaskDistribution_experiment_2 

# daggit init dag_examples/task_distribution/task_distribution_3.yaml &
# daggit run TaskDistribution_experiment_3 &

# daggit init dag_examples/task_distribution/task_distribution_4.yaml &
# daggit run TaskDistribution_experiment_4 &

# daggit init dag_examples/task_distribution/task_distribution_5.yaml &
# daggit run TaskDistribution_experiment_5 &

# daggit init dag_examples/task_distribution/task_distribution_6.yaml &
# daggit run TaskDistribution_experiment_6 &

# daggit init dag_examples/task_distribution/task_distribution_7.yaml &
# daggit run TaskDistribution_experiment_7 &

# daggit init dag_examples/task_distribution/task_distribution_8.yaml &
# daggit run TaskDistribution_experiment_8 &
# pybuilderInstalled=`pip freeze | grep 'pybuilder' | wc -l`

# if [ $pybuilderInstalled != 1 ]
# then
#    echo "Installing pybuilder"
#    pip install pybuilder
# fi

# pyb install_dependencies clean publish
# pyb sphinx_generate_documentation
# tox

# rm -rf mlwb_venv/

# if [ ! -d "bin" ]; then
#   mkdir 'bin'
# fi

# cp target/dist/daggit-0.5.0/dist/* bin/

# #rm -rf target/
