from lib.operation_definition import NodeOperation

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier


def sklearn_model(node, func):

    model = NodeOperation(node) #initiate

    # check if i/o expectations are met
    model.io_check((len(model.inputs) == 2) & (len(model.outputs)== 3))

    [train, test] = model.get_dataframes(model.inputs)  # get input data
    target_col = model.arguments['target']

    # do something
    train_X = train[train.columns[train.columns.values != target_col]]
    train_y = train[target_col].squeeze()
    test_X = test[test.columns[test.columns.values != target_col]]
    test_y = test[target_col].squeeze()

    # method definition
    method = func
    method.fit(train_X, train_y)
    prediction = method.predict(test_X)
    prediction = pd.DataFrame(prediction, columns=['predictions']) # ALL DATA OBJECTS STORED MUST BE DATAFRAMES

    # Save model, prediction, report
    with open(model.get_location(model.outputs[0],'file'), 'wb') as handle:
        pickle.dump(prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model.put_dataframes([prediction], [model.outputs[1]])

    with open(model.get_location(model.outputs[2],'file'), "w") as text_file:
        text_file.write('The accuracy of {} model is {}'
                        .format(type(func), metrics.accuracy_score(prediction, test_y)))


def logistic_model(node):
    sklearn_model(node, LogisticRegression())


def svm_model(node):
    sklearn_model(node, svm.SVC())


def decision_tree(node):
    sklearn_model(node, DecisionTreeClassifier())