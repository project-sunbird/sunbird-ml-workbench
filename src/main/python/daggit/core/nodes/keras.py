import importlib
import pandas as pd
from daggit.core.base.factory import BaseOperator
from daggit.core.io.io import Pandas_Dataframe, Pickle_Obj, File_Txt
from daggit.core.nodes.registry import get_node_callable


def baseline_model_svm():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, input_dim=8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(
        loss='categorical_hinge',
        optimizer='adam',
        metrics=['accuracy'])
    return model


class train_svm_model(BaseOperator):

    @property
    def inputs(self):
        return {"train": Pandas_Dataframe(self.node.inputs[0]),
                "test": Pandas_Dataframe(self.node.inputs[1])}

    @property
    def outputs(self):
        return {"model": File_Txt(self.node.outputs[0]),
                "model_weights": Pickle_Obj(self.node.outputs[1]),
                "predictions": Pandas_Dataframe(self.node.outputs[2]),
                "report": Pickle_Obj(self.node.outputs[3]),
                }

    def run(
            self,
            target_variable,
            ignore_variables=None,
            model_args=None,
            cv_args=None):
        from keras.utils import np_utils
        from sklearn.preprocessing import LabelEncoder

        train = self.inputs["train"].read().drop(ignore_variables, axis=1)
        test = self.inputs["test"].read().drop(ignore_variables, axis=1)

        if ignore_variables is not list:
            ignore_variables = [ignore_variables]

        train_y = train[target_variable]
        train_X = train.drop(target_variable, axis=1).values

        test_y = test[target_variable]
        test_X = test.drop(target_variable, axis=1).values

        imports = list(self.node.imports)
        for importlist in imports:
            module_name = importlist.pop(0)
            for imp in importlist:
                importlib.import_module(name=module_name, package=imp)
        print(self.node.imports)
        model = get_node_callable(
            module_path=model_args['module_path'],
            operator=model_args['name'])
        model_params = model_args['arguments']

        encoder = LabelEncoder()
        encoder.fit(train_y)
        encoded_Y = encoder.transform(train_y)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = np_utils.to_categorical(encoded_Y)

        encoded_Y_test = encoder.transform(test_y)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y_test = np_utils.to_categorical(encoded_Y_test)
        estimator = model(build_fn=baseline_model_svm, **model_params)

        estimator.fit(x=train_X, y=dummy_y)

        model_score = estimator.score(x=test_X, y=dummy_y_test)
        pred = estimator.predict(x=test_X)
        predictions = pd.DataFrame(
            encoder.inverse_transform(pred),
            columns=['predictions'])
        report = 'The accuracy of the model is: ' + str(model_score)

        model_json = estimator.model.to_json()
        model_weights = estimator.model.get_weights()

        self.outputs["model"].write(model_json)
        self.outputs["model_weights"].write(model_weights)
        self.outputs["predictions"].write(predictions)
        self.outputs["report"].write(report)
