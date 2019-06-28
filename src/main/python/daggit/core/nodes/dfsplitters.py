from sklearn.model_selection import KFold
from src.main.python.daggit.core.io.io import Pandas_Dataframe
from src.main.python.daggit.core.base.factory import BaseOperator


class k_fold_splitter(BaseOperator):

    @property
    def inputs(self):
        return {"input_df": Pandas_Dataframe(self.node.inputs[0])
                }

    @property
    def outputs(self):
        return {"train": Pandas_Dataframe(self.node.outputs[0]),
                "test": Pandas_Dataframe(self.node.outputs[1])}

    def run(
            self,
            drop_missing_perc,
            target_variable,
            ignore_variables,
            categorical_impute,
            numeric_impute):
        input_df = self.inputs["input_df"].read()

        kf = KFold(**self.node.arguments)
        for train_index, test_index in kf.split(input_df):
            data_train = input_df.iloc[train_index]
            data_test = input_df.iloc[test_index]
        self.outputs["train"].write(data_train)
        self.outputs["test"].write(data_test)
