from future.builtins import super
from airflow.operators.python_operator import PythonOperator
from daggit.oplib.operators_registry import get_op_callable


class DaggitPyOp(PythonOperator):

    def __init__(self, node, dag, *args, **kwargs):

        task_id = node.task_id

        # do import, search and load
        concrete_op_object = get_op_callable(node.operation)(node)
        python_callable = getattr(concrete_op_object, '_run')
        op_kwargs = node.arguments
        super().__init__(task_id= task_id, python_callable=python_callable, op_kwargs= op_kwargs, dag=dag,
                          *args, **kwargs)
