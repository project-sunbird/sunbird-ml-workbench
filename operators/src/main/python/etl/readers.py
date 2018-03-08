import pandas as pd
from core.lib.operation_definition import NodeOperation


def csv_reader(node):
    reader = NodeOperation(node) # initiate
    reader.io_check(len(reader.inputs) == len(reader.outputs)) # check if i/o expectations are met
    reader.read_files_to_dataframe_storage(pd.read_csv) # pass function to read files