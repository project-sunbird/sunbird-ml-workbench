from mlworkbench.utils.common import dicts_union
from mlworkbench.lib.etl.readers import csv_reader
from mlworkbench.lib.etl.writers import csv_writer
from mlworkbench.lib.etl.splitters import k_fold_splitter
from mlworkbench.lib.models.scikit import logistic_model, svm_model, decision_tree

python_callables = {
    'csv_reader': csv_reader,
    'csv_writer' : csv_writer,
    'k_fold_splitter': k_fold_splitter,
    'logistic_model': logistic_model,
    'svm_model': svm_model,
    'decision_tree': decision_tree,
}

scala_spark_callables = {
}

callables = dicts_union(python_callables, scala_spark_callables)

def register(operator):
    if(operator.getType() == 'python'):
        python_callables[operator.getId] = operator
    if(operator.getType() == 'scala'):
        scala_spark_callables[operator.getId] = operator