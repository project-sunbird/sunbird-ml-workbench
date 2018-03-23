from mlworkbench.utils.common import dicts_union
from mlworkbench.lib.etl.readers import csv_reader
from mlworkbench.lib.etl.writers import csv_writer
from mlworkbench.lib.etl.splitters import k_fold_splitter, df_splitter_on_column_values
from mlworkbench.lib.models.scikit import logistic_model, svm_model, decision_tree
from mlworkbench.lib.custom.custom_operators_registry import custom_python_callables, custom_bash_callables


python_callables = {
    'csv_reader': csv_reader,
    'csv_writer' : csv_writer,
    'k_fold_splitter': k_fold_splitter,
    'df_splitter_on_column_values': df_splitter_on_column_values,
    'logistic_model': logistic_model,
    'svm_model': svm_model,
    'decision_tree': decision_tree
}

bash_callables = {
}

python_callables = dicts_union(python_callables,custom_python_callables)
bash_callables = dicts_union(bash_callables, custom_bash_callables)

callables = dicts_union(python_callables, bash_callables)
