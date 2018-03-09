from mlworkbench.lib.operation_definition import NodeOperation
from sklearn.model_selection import KFold


def k_fold_splitter(node):

    splitter = NodeOperation(node) # initiate

    # check if i/o expectations are met
    splitter.io_check((len(splitter.inputs) == 1) & (len(splitter.outputs) == 2*splitter.arguments['n_splits']))

    data = splitter.get_dataframes(splitter.inputs)[0] # get input data

    # do something
    kf = KFold(**splitter.arguments)

    df_list = []
    for train_index, test_index in kf.split(data):
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]
        df_list.extend([data_train, data_test])

    # store output
    splitter.put_dataframes(df_list,splitter.outputs)