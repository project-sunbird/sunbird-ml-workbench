#!/bin/sh
# Assumptions
# 1. python 2.7 is installed

# 1. Install python virtual env
echo "=====> Installing MLWorkbench..."

echo "=====> Installing virtual environment..."
pip install virtualenv
virtualenv venv
source venv/bin/activate

# 2. Install dependencies
echo "=====> Installing dependencies..."
pip install numpy
pip install airflow==1.8.0 pandas sklearn pint tables networkx findspark scipy surprise
cp /Users/santhosh/github-repos/ML-Workbench/dist/mlworkbench-0.0.1.tar.gz .
tar xvzf mlworkbench-0.0.1.tar.gz
cd mlworkbench-0.0.1

echo "=====> Installing MLWorkbench core..."
python ./setup.py install

cd ..

rm -rf workingdir
mkdir -p workingdir/airflow
mkdir -p workingdir/airflow_dag_executor

cp mlworkbench-0.0.1/mlworkbench/executor/airflow_executor/dags/airflow_dag.py workingdir/airflow_dag_executor

#TODO: Clean up mlworkbench-0.0.1 folder and zip file
rm -rf mlworkbench-0.0.1

echo "=====> MLWorkbench installation complete!!!"