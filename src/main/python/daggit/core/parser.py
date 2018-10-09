import six
import yaml
import os
from .config import STORE
from ..core.base import Node
from ..core.utils import get_as_list, normalize_path


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
            graph_inputs[key] = normalize_path(cwd=os.path.dirname(dag_config_file), path= dag_config["inputs"][key])
    else:
        graph_inputs = dag_config["inputs"]

    graph_outputs = {}
    if STORE.lower() == 'local':
        for key in dag_config["outputs"].keys():
            graph_outputs[key] = normalize_path(cwd=os.path.dirname(dag_config_file), path=dag_config["outputs"][key])
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
            inputs_parents[".".join([task,label])] = task

    nodes_bag = {}
    for node_config in graph_config:
        node = Node(node_config, experiment_name, owner, graph_inputs, graph_outputs, inputs_parents)
        nodes_bag[node.task_id] = node

    return nodes_bag
