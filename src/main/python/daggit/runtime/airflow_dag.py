import six
import os
from airflow import DAG
from datetime import datetime, date, timedelta
from daggit.core.parser import get_nodes
from daggit.runtime.airflow_runtime import DaggitPyOp


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'graph_location')

f = open(filename, "r")

nodes = get_nodes(f.read())

now = date.today()
day_before = now - timedelta(2)

sample_node = list(nodes.values())[0]
default_args = {
    'owner': sample_node.owner,
    'depends_on_past': False,
    'start_date': datetime(day_before.year, day_before.month, day_before.day),
}

dag = DAG(sample_node.dag_id, default_args=default_args, schedule_interval='@once')

nodes_upstream = {}
for node in list(nodes.values()):
    locals()[node.task_id] = DaggitPyOp(node=node, dag=dag)
    upstream_tasks = []
    for i in node.inputs:
        upstream_tasks.append(i.parent_task)
    nodes_upstream[node.task_id] = list(set(upstream_tasks))


for node, upstream_list in six.iteritems(nodes_upstream):
    upstream_list = [e for e in upstream_list if e is not None]
    if len(upstream_list) > 0:
        for upstream_task in upstream_list:
            locals()[node].set_upstream(locals()[upstream_task])



