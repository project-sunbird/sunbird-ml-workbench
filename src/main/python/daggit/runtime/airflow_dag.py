import os
from daggit.core.base.parser import get_dag

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'graph_location')

f = open(filename, "r")
for index, dag in enumerate(f):
    file = dag.replace('\n', '')
    dag_name = file[:-5].split("/")[-1] + str(index)
    globals()[dag_name] = get_dag(file)
