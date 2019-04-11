from sklearn.pipeline import Pipeline

from daggit.core.io.io import Pandas_Dataframe, File_Txt, Pickle_Obj
from daggit.core.base.factory import BaseOperator
from daggit.core.nodes.registry import get_node_callable
from daggit.core.oplib.etl import DFFeatureUnion, ColumnExtractor
from daggit.core.oplib.etl import DFMissingStr, DFOneHot
from daggit.core.oplib.etl import DFMissingNum


class CustomPreprocess(BaseOperator):

    @property
    def inputs(self):
        return {"train": Pandas_Dataframe(self.node.inputs[0]),
                "test": Pandas_Dataframe(self.node.inputs[1])}

    @property
    def outputs(self):
        return {"preprocessed_train": Pandas_Dataframe(self.node.outputs[0]),
                "preprocessed_test": Pandas_Dataframe(self.node.outputs[1])}

    def run(
            self,
            drop_missing_perc,
            target_variable,
            ignore_variables,
            categorical_impute,
            numeric_impute):
        train = self.inputs["train"].read()
        test = self.inputs["test"].read()

        if ignore_variables is not list:
            ignore_variables = [ignore_variables]

        data_availability = train.describe(
            include='all').loc['count'] / train.shape[0]
        selected_cols = data_availability[data_availability >
                                          drop_missing_perc].index
        selected_cols = set(selected_cols) - \
            (set([target_variable]).union(set(ignore_variables)))

        numeric_cols = list(
            set(list(train._get_numeric_data())).intersection(selected_cols))
        categorical_cols = list(selected_cols - set(numeric_cols))

        preprocess = Pipeline([("features",
                                DFFeatureUnion([("numeric",
                                                 Pipeline([("num_sel",
                                                            ColumnExtractor(numeric_cols)),
                                                           ("num_impute",
                                                            DFMissingNum(replace='median'))])),
                                                ("categorical",
                                                 Pipeline([("cat_sel",
                                                            ColumnExtractor(categorical_cols)),
                                                           ("str_impute",
                                                            DFMissingStr(replace='most_frequent')),
                                                           ("one_hot",
                                                            DFOneHot())]))]))])

        processed_train = preprocess.fit_transform(train)
        processed_train[target_variable] = train[target_variable]
        processed_test = preprocess.transform(test)

        self.outputs["preprocessed_train"].write(processed_train)
        self.outputs["preprocessed_test"].write(processed_test)


class CrossValidate(BaseOperator):

    @property
    def inputs(self):
        return {"preprocessed_train": Pandas_Dataframe(self.node.inputs[0])}

    @property
    def outputs(self):
        return {
            "report": File_Txt(
                self.node.outputs[0]), "model": Pickle_Obj(
                self.node.outputs[1])}

    def run(self, target_variable, model_args, cv_args):
        from sklearn.model_selection import cross_val_score
        preprocessed_train = self.inputs['preprocessed_train'].read()

        y = preprocessed_train[target_variable]
        X = preprocessed_train.drop(target_variable, axis=1).values

        imports = self.node.imports
        model = get_node_callable(imports[0][1], module_path=imports[0][0])

        model_params = model_args['arguments']
        reg = model(**model_params)
        scores = cross_val_score(reg, X, y, **cv_args)

        reg.fit(X, y)
        report = "MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

        self.outputs["model"].write(model)
        self.outputs["report"].write(report)


class Splitters(BaseOperator):
    @property
    def inputs(self):
        return {"input_df": Pandas_Dataframe(self.node.inputs[0])
                }

    @property
    def outputs(self):
        return {"train": Pandas_Dataframe(self.node.outputs[0]),
                "test": Pandas_Dataframe(self.node.outputs[1])}

    def run(self, model_args):
        input_df = self.inputs['input_df'].read()

        imports = self.node.imports
        model = get_node_callable(imports[0][1], module_path=imports[0][0])

        model_params = model_args['arguments']

        train, test = model(input_df, **model_params)
        self.outputs["train"].write(train)
        self.outputs["test"].write(test)
