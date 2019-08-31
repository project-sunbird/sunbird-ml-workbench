import importlib
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(os.path.join(this_dir, '../..'))


def get_node_callable(operator, module_path=None): #get_op_callable
    if module_path:
        module = importlib.import_module(module_path)
        return getattr(module, operator)
    else:
        operator_full_path = operator.split('.')
        operator_name = operator_full_path.pop(-1)
        operator_module = '.'.join(operator_full_path)
        module_path_list = []
        for root, dirs, _ in os.walk(abs_path):
            if len(dirs) > 0:
                if "nodes" in dirs:                    
                    operator_dir = os.path.join(root, "nodes")
                    module_path_list.append(
                        operator_dir[operator_dir.rfind("daggit"):].replace("/", "."))
        for path in module_path_list:
            try:
                module = importlib.import_module(path + '.' + operator_module)
            except BaseException:
                continue
        return getattr(module, operator_name)
