#!/bin/sh
# Assumptions
# 1. python 2.7 is installed

# 1. Install python virtual env
echo "=====> Installing virtual environment..."
pip install virtualenv > /dev/null 2>&1
virtualenv venv > /dev/null 2>&1
source venv/bin/activate

# 2. Install dependencies
echo "=====> Installing dependencies..."
pip install numpy
pip install airflow==1.8.0 pandas sklearn pint tables networkx findspark scipy surprise
cp /Users/santhosh/github-repos/ML-Workbench/dist/mlworkbench-0.0.1.tar.gz . > /dev/null 2>&1
tar xvzf mlworkbench-0.0.1.tar.gz > /dev/null 2>&1
cd mlworkbench-0.0.1

echo "=====> Installing mlworkbench..."
python ./setup.py install

# 3. Setup airflow home directory
rm -rf workingdir
mkdir workingdir

cwd=`pwd`
echo "Current Dir: $pwd"

# 4. Move the parser to airflow_home directory