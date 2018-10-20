import importlib
# from .core.sklearn_lib import CustomPreprocess, CrossValidate
# from .core.df_splitters import k_fold_splitter
#
# operators_list = {
#     "CustomPreprocess" : CustomPreprocess,
#     "CrossValidate" : CrossValidate,
#     "k_fold_splitter": k_fold_splitter
# }


def get_op_callable(operator, module_path=None):
    # improve the search path later
    if (module_path):
        module = importlib.import_module(module_path)
        return getattr(module, operator)
    else:
        try:
            module_path = 'daggit.core.operators'
            operator_full_path = operator.split('.')
            operator_name = operator_full_path.pop(-1)
            operator_module = '.'.join(operator_full_path)
            module = importlib.import_module(module_path + '.' + operator_module)
        except:
            module_path = 'daggit.contrib.ekstep.operators'
            operator_full_path = operator.split('.')
            operator_name = operator_full_path.pop(-1)
            operator_module = '.'.join(operator_full_path)
            module = importlib.import_module(module_path + '.' + operator_module)

    return getattr(module, operator_name)

