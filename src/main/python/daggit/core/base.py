from abc import ABCMeta, abstractmethod
from ..core.utils import get_as_list

class BaseOperator(object):
    __metaclass__ = ABCMeta

    def __init__(self, node):
        self.node = node

    @property
    @abstractmethod
    def inputs(self):
        raise NotImplementedError('Needs to be implemented!')

    @property
    @abstractmethod
    def outputs(self):
        raise NotImplementedError('Needs to be implemented!')

    @abstractmethod
    def run(self, **kwargs):
        raise NotImplementedError('Needs to be implemented!')

    def _run(self, **kwargs):
        self._pre_execute()
        self.run(**kwargs)

    def _pre_execute(self):
        pass # TODO


class BaseDAGManager(object):
    __metaclass__ = ABCMeta

    def __init__(self, dag):
        self.dag = dag

    @abstractmethod
    def init(self):
        raise NotImplementedError('dagit init Needs to be implemented!')

    @abstractmethod
    def run(self):
        raise NotImplementedError('dagit run needs to be implemented!')


class NodeData(object):

    def __init__(self, dag_id, task_id, data_alias, parent_task=None, external_ref=None):
        self.dag_id = dag_id
        self.task_id = task_id
        self.data_alias = data_alias
        self.parent_task = parent_task
        self.external_ref = external_ref

class Node(object):

    def __init__(self, node_config, experiment_name, owner, graph_inputs, graph_outputs, inputs_parents):

        self.dag_id = experiment_name
        self.owner = owner
        self.graph_inputs = graph_inputs
        self.graph_outputs = graph_outputs
        self.task_id = node_config["node_name"]
        self.inputs = get_as_list(node_config["inputs"])
        self.outputs = get_as_list(node_config["outputs"])
        self.operation = node_config["operation"]
        self.inputs_parents = inputs_parents

        if "imports" in node_config:
            self.imports = node_config["imports"]
        else:
            self.imports = None

        if "arguments" in node_config:
            self.arguments = node_config["arguments"]
        else:
            self.arguments = {}

        self.infer_data_objects()

    def infer_data_objects(self):
        
        inputs_list = []
        for data_label in self.inputs:

            if data_label in self.graph_inputs:
                data_object = NodeData(dag_id=self.dag_id, task_id=self.task_id,
                                   data_alias=data_label, external_ref=self.graph_inputs[data_label])
            else:
                try:
                    data_name = data_label.split(".")[1]
                except IndexError:
                    data_name = data_label.split(".")[0]
                data_object = NodeData(dag_id=self.dag_id, task_id=self.task_id, data_alias=data_name,
                                   parent_task=self.inputs_parents[data_label])

            inputs_list.append(data_object)
        self.inputs = inputs_list

        outputs_list = []
        for data_label in self.outputs:
            namespaced_data_label = self.task_id+'.'+data_label
            if namespaced_data_label in self.graph_outputs:
                data_object = NodeData(dag_id=self.dag_id, task_id=self.task_id, parent_task=self.task_id,
                                       data_alias=data_label, external_ref=self.graph_outputs[namespaced_data_label])
            elif data_label in self.graph_outputs:
                data_object = NodeData(dag_id=self.dag_id, task_id=self.task_id, parent_task=self.task_id,
                                       data_alias=data_label, external_ref=self.graph_outputs[data_label])
            else:
                data_object = NodeData(dag_id=self.dag_id, task_id=self.task_id, data_alias=data_label,
                                       parent_task=self.task_id)

            outputs_list.append(data_object)
        self.outputs = outputs_list
