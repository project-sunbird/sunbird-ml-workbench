#!/bin/sh

source venv/bin/activate

rm -rf workingdir/dags
mkdir -p workingdir/dags

cp $1 workingdir/dags

export AIRFLOW_HOME="`pwd`/workingdir/airflow"
export AIRFLOW__CORE__DAGS_FOLDER="`pwd`/workingdir/airflow_dag_executor/"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"
export DAG_FOLDER="`pwd`/workingdir/dags"
export MLWB_CWD="`pwd`"

experimentName=`grep 'experiment_name:' $1 | awk '{print $2}'`

airflow initdb
airflow clear -c $experimentName
airflow unpause $experimentName

airflow scheduler --pid scheduler.pid -d $experimentName