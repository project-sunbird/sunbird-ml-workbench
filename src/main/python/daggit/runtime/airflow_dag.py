import six
import os
import yaml

from airflow import DAG
from datetime import datetime
from daggit.core.base.config import STORE
from daggit.core.base.factory import Node
from daggit.core.base.utils import get_as_list, normalize_path
from daggit.runtime.airflow_runtime import DaggitPyOp
from daggit.core.base.parser import create_dag

#add b'DAG', b'airflow' to script

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'graph_location')

f = open(filename, "r")
for index, dag in enumerate(f):
    file = dag.replace('\n', '')
    dag_name = file[:-5].split("/")[-1] + str(index)
    globals()[dag_name] = create_dag(file)
