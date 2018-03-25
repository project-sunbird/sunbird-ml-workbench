# -*- coding: utf-8 -*-
#
# Copyright (C) 2016  EkStep Foundation
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import os
import glob
from os.path import expanduser

#airflow_dir = os.environ["AIRFLOW_HOME"]
#sys.path.insert(0,os.path.join(airflow_dir,os.pardir))

from mlworkbench.lib.operators_registry import python_callables, bash_callables
from mlworkbench.executor.parse_configuration import create_nodes, createDAG

from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator

if "DAG_FOLDER" in os.environ:
    working_dir = os.environ["DAG_FOLDER"]
else:
    raise ValueError("Cannot find the dags folder to register the dags")

path = working_dir + '/'
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
        if node.operation in bash_callables:
            bash_script = bash_callables[node.operation](node)
            exec (
                "{} = BashOperator(task_id=node.node_name, bash_command=bash_script, dag = dag_{})"
                    .format(node.node_name, i))

        # Airflow construction of graph - assign parents
        if node.parents != []:
            for parent in node.parents:
                exec ("{} >> {}".format(parent, node.node_name))
