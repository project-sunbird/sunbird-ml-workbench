import six
import yaml
import os
from airflow import DAG


from datetime import datetime
from daggit.core.base.config import STORE
from daggit.core.base.config import ORCHESTRATOR_AIRFLOW_dag_config_depends_on_past
from daggit.core.base.config import ORCHESTRATOR_AIRFLOW_dag_config_schedule_interval
from daggit.core.base.config import ORCHESTRATOR_AIRFLOW_dag_config_start_date
from daggit.core.base.factory import Node
from daggit.core.base.utils import get_as_list, normalize_path
from daggit.runtime.airflow_runtime import DaggitPyOp


def get_nodes(dag_config_file):
    dag_config = {}

    # Load dag configuration
    with open(dag_config_file, 'r', encoding="latin1") as stream:
        try:
            dag_config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    graph_inputs = {}
    if STORE.lower() == 'local':
        for key in dag_config["inputs"].keys():
            graph_inputs[key] = normalize_path(
                cwd=os.path.dirname(dag_config_file),
                path=dag_config["inputs"][key])
    else:
        graph_inputs = dag_config["inputs"]

    graph_outputs = {}
    if STORE.lower() == 'local':
        for key in dag_config["outputs"].keys():
            graph_outputs[key] = normalize_path(
                cwd=os.path.dirname(dag_config_file),
                path=dag_config["outputs"][key])
    else:
        graph_outputs = dag_config["outputs"]

    experiment_name = dag_config["experiment_name"]
    owner = dag_config["owner"]
    # graph_inputs = dag_config["inputs"]
    # graph_outputs = dag_config["outputs"]
    graph_config = dag_config["graph"]

    inputs_parents = {}
    for node_config in graph_config:
        task = node_config['node_name']
        outputs = get_as_list(node_config['outputs'])
        for label in outputs:
            inputs_parents[".".join([task, label])] = task

    nodes_bag = {}
    for node_config in graph_config:
        node = Node(
            node_config,
            experiment_name,
            owner,
            graph_inputs,
            graph_outputs,
            inputs_parents)
        nodes_bag[node.task_id] = node

    return nodes_bag


def create_dag(dag_config_file):
    dag_config = {}
    with open(dag_config_file, 'r', encoding="latin1") as stream:
        try:
            dag_config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    experiment_name = dag_config["experiment_name"]
    owner = dag_config["owner"]

    default_args = {}

    # depends_on_past
    try:
        default_args['depends_on_past'] = dag_config['dag_config']['depends_on_past']
    except BaseException:
        default_args['depends_on_past'] = ORCHESTRATOR_AIRFLOW_dag_config_depends_on_past

    # start_date
    datepattern = '%d-%m-%Y'
    try:
        default_args['start_date'] = datetime.strptime(
            dag_config['dag_config']['start_date'], datepattern)
    except BaseException:
        default_args['start_date'] = datetime.strptime(
            ORCHESTRATOR_AIRFLOW_dag_config_start_date, datepattern)

    # schedule_interval
    try:
        schedule_interval = dag_config['dag_config']['schedule_interval']
    except BaseException:
        schedule_interval = ORCHESTRATOR_AIRFLOW_dag_config_schedule_interval

    default_args['owner'] = owner
    print("Experiment name: ", experiment_name)
    dag = DAG(
        experiment_name,
        default_args=default_args,
        schedule_interval=schedule_interval)

    graph_inputs = {}
    if STORE.lower() == 'local':
        for key in dag_config["inputs"].keys():
            graph_inputs[key] = normalize_path(
                cwd=os.path.dirname(dag_config_file),
                path=dag_config["inputs"][key])
    else:
        graph_inputs = dag_config["inputs"]

    graph_outputs = {}
    if STORE.lower() == 'local':
        for key in dag_config["outputs"].keys():
            graph_outputs[key] = normalize_path(
                cwd=os.path.dirname(dag_config_file),
                path=dag_config["outputs"][key])
    else:
        graph_outputs = dag_config["outputs"]

    # graph_inputs = dag_config["inputs"]
    # graph_outputs = dag_config["outputs"]
    graph_config = dag_config["graph"]

    inputs_parents = {}
    for node_config in graph_config:
        task = node_config['node_name']
        outputs = get_as_list(node_config['outputs'])
        for label in outputs:
            inputs_parents[".".join([task, label])] = task

    nodes_bag = {}
    for node_config in graph_config:
        node = Node(
            node_config,
            experiment_name,
            owner,
            graph_inputs,
            graph_outputs,
            inputs_parents)
        nodes_bag[node.task_id] = node

    # Dag creation
    nodes_upstream = {}
    for node in list(nodes_bag.values()):
        globals()[node.task_id] = DaggitPyOp(node=node, dag=dag)
        upstream_tasks = []
        for i in node.inputs:
            upstream_tasks.append(i.parent_task)
        nodes_upstream[node.task_id] = list(set(upstream_tasks))

    for node, upstream_list in six.iteritems(nodes_upstream):
        upstream_list = [e for e in upstream_list if e is not None]
        if len(upstream_list) > 0:
            for upstream_task in upstream_list:
                globals()[node].set_upstream(globals()[upstream_task])

    return dag
