import os

from airflow.models import DAG
from daggit.core.base.parser import create_dag

# DO NOT REMOVE THE COMMENT
# add b'DAG', b'airflow' to script

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'graph_location')

f = open(filename, "r")
lines = f.read().splitlines()
last_yaml = lines[-1]
dag_name = os.path.basename(last_yaml)[:-5] + str(0)
globals()[dag_name] = create_dag(last_yaml)

# for index, dag in enumerate(f):
#     file = dag.replace('\n', '')
#     dag_name = os.path.basename(file)[:-5] + str(index)
#     # dag_name = file[:-5].split("/")[-1] + str(index)
#     globals()[dag_name] = create_dag(file)
