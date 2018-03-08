'''
Functions to parse YAML configuration and create DAG
'''

import yaml
import numpy as np
from datetime import datetime, date, timedelta
from lib.node_registry import callables
from utils.common import create_directory, dir_loc
from collections import OrderedDict
import networkx as nx
import pickle
from airflow import DAG

class NodeObject:

    def __init__(self, node):

        # node properties
        self.node_name = node['node_name']
        self.parents = []
        self.operation = node['operation']
        self.arguments = node['arguments']
        self.inputs = node['inputs']
        self.outputs = node['outputs']
        self.experiment_dir = node['experiment_dir']
        self.graph_inputs = node['graph_inputs']
        self.graph_outputs = node['graph_outputs']

        if type(self.inputs) == str:
            self.inputs = [self.inputs]
        if type(self.outputs) == str:
            self.outputs = [self.outputs]

        if self.inputs is None:
            self.inputs = []
        if self.outputs is None:
            self.outputs = []
        if self.graph_inputs is None:
            self.graph_inputs = {}
        if self.graph_outputs is None:
            self.graph_outputs = {}

    def print_node(self):
        print_obj = {}
        print_obj['node_name'] = self.node_name
        print_obj['parents'] = self.parents
        print_obj['operation'] = self.operation
        print_obj['arguments'] = self.arguments
        print_obj['inputs'] = self.inputs
        print_obj['outputs'] = self.outputs

        print print_obj


class ComplexNodeObject:

    @staticmethod
    def _node_arguments_simplifier(properties, index):
        if type(properties) is list:
            out = properties[index]
        else:
            out = properties
        return out

    def __init__(self, node):

        self.nodes = {}
        for i in range(0, len(node['node_name'])):
            node_properties = {}
            node_properties['node_name'] = self._node_arguments_simplifier(node['node_name'], i)
            node_properties['operation'] = self._node_arguments_simplifier(node['operation'], i)
            node_properties['arguments'] = self._node_arguments_simplifier(node['arguments'], i)
            node_properties['inputs'] = self._node_arguments_simplifier(node['inputs'], i)
            node_properties['outputs'] = self._node_arguments_simplifier(node['outputs'], i)
            node_properties['experiment_dir'] = node['experiment_dir']
            node_properties['graph_inputs'] = node['graph_inputs']
            node_properties['graph_outputs'] = node['graph_outputs']
            self.nodes[node_properties['node_name']] = NodeObject(node_properties)


def __assign_parents(org_nodes_bag):

    nodes_bag = org_nodes_bag.copy()

    # get parents of the nodes
    for node_name, node in nodes_bag.iteritems():
        for other_node_name, other_node in nodes_bag.iteritems():
            if other_node_name != node_name:
                if len(set(node.inputs)& set(other_node.outputs)) > 0:
                    node.parents.append(other_node.node_name)
    return nodes_bag


def _create_nodes_bag(config_loc):
    learning_config = {}

    with open(config_loc, 'r') as stream:
        try:
            learning_config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    nodes = learning_config['graph']

    nodes_bag = {}

    for i in range(0, len(nodes)):
        node_properties = nodes[i]
        node_properties['experiment_dir'] = learning_config['experiment_dir']
        node_properties['graph_inputs'] = learning_config['inputs']
        node_properties['graph_outputs'] = learning_config['outputs']

        if type(node_properties['node_name']) == list:
            complex_node = ComplexNodeObject(node_properties)
            nodes_bag.update(complex_node.nodes)
        else:
            node = NodeObject(node_properties)
            nodes_bag[node.node_name] = node

    nodes_bag = __assign_parents(nodes_bag)

    return nodes_bag


def create_nodes(config_loc):

    learning_config = {}

    # Load learning configuration
    with open(config_loc, 'r') as stream:
        try:
            learning_config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # links in graph definition
    experiment_dir = learning_config['experiment_dir']
    graph_inputs = learning_config['inputs']

    if graph_inputs is None:
        graph_inputs = {}

    # create experiment directory if it doesn't exist
    create_directory(experiment_dir)

    # get nodes from config
    nodes = _create_nodes_bag(config_loc)

    graph = nx.DiGraph()
    edge_list = []
    output_node = {}  # store mapping between output objects and nodes

    # Go through the nodes bag and make sure everything is okay
    for node in nodes.values():

        # Make sure node is valid
        # - operator exists
        if node.operation not in callables.keys():
            raise ValueError("{} node contains unrecognised operation {}.".format(node.node_name, node.operation))

        # - inputs are mapped to parent or graph inputs
        all_inputs = set(graph_inputs.keys())

        for parent in node.parents:
            all_inputs = all_inputs.union(set(nodes[parent].outputs))

        if not (set(node.inputs).issubset(all_inputs)):
            raise ValueError("{} node contains unrecognised input object(s).".format(node.node_name))

        # create edges to construct the graph
        sink = node.node_name
        graph.add_node(sink)
        for src in node.parents:
            edge_list.append([src, sink])

        # store mapping of nodes to ouputs
        for output_name in node.outputs:
            output_node[output_name] = node.node_name

    # Make sure its a DAG
    graph.add_edges_from(edge_list) # create DAG
    if not(nx.is_directed_acyclic_graph(graph)): #check
        raise ValueError("Graph is not a valid DAG.")

    # sort nodes dict by nodes sequence
    nodes_sequence = list(nx.topological_sort(graph))
    nodes_sorted = OrderedDict()
    for node_name in nodes_sequence:
        nodes_sorted[node_name] = nodes[node_name]

    # save output - node mapping in experiment dir
    np.save(dir_loc(experiment_dir)+'output_node_mapping.npy',output_node)

    with open(dir_loc(experiment_dir)+'graph.pkl', 'wb') as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return nodes_sorted


def createDAG(config_loc):
    learning_config = {}

    # Load learning configuration
    with open(config_loc, 'r') as stream:
        try:
            learning_config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    now = date.today()
    day_before = now - timedelta(2)

    # set start date 2 days before today to ensure it always executes

    default_args = {
        'owner': learning_config['owner'],
        'depends_on_past': False,
        'start_date': datetime(day_before.year, day_before.month, day_before.day),
        #'start_date': datetime(2018,2,5)
    }

    dag = DAG(learning_config['experiment_name'], default_args=default_args, schedule_interval='@once')

    return dag