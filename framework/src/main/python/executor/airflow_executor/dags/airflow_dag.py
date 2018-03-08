import sys
import os
import glob
from os.path import expanduser

airflow_dir = os.environ["AIRFLOW_HOME"]
sys.path.insert(0,os.path.join(airflow_dir,os.pardir,os.pardir,os.pardir))

from lib.node_registry import python_callables, scala_spark_callables
from executor.parse_configuration import create_nodes, createDAG

from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator

if "MLWB_HOME" in os.environ:
    working_dir = os.environ["MLWB_HOME"]
else:
    working_dir = os.path.join(os.path.dirname(__file__),os.pardir,os.pardir
                               ,os.pardir,os.pardir,"examples")

path = os.path.join(working_dir, 'configurations') + '/'
extension = 'yaml'
os.chdir(path)
config_files = [i for i in glob.glob('*.{}'.format(extension))]

for i in range(0, len(config_files)):

    config_loc = path + config_files[i]

    # Create DAG
    exec ("dag_{} = createDAG(config_loc)".format(i))

    # Create nodes from config
    nodes = create_nodes(config_loc)

    for node in nodes.values():

        # Create Airflow Tasks

        # python operator
        if node.operation in python_callables:
            exec (
                "{} = PythonOperator(task_id=node.node_name, python_callable=python_callables[node.operation], op_kwargs={{'node': node}}, dag = dag_{})"
                    .format(node.node_name, i))

            # bash operator
        if node.operation in scala_spark_callables:
            bash_script = scala_spark_callables[node.operation](node)
            exec (
                "{} = BashOperator(task_id=node.node_name, bash_command=bash_script, dag = dag_{})"
                    .format(node.node_name, i))

        # Airflow construction of graph - assign parents
        if node.parents != []:
            for parent in node.parents:
                exec ("{} >> {}".format(parent, node.node_name))