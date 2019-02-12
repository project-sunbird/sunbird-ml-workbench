import time
import os

from daggit.core.io.io import Pandas_Dataframe, File_Txt
from daggit.core.io.io import ReadDaggitTask_Folderpath
from daggit.core.base.factory import BaseOperator

class Task1(BaseOperator):

    @property
    def inputs(self):
        return {"DS_DATA_HOME": ReadDaggitTask_Folderpath(
                self.node.inputs[0])
                }

    @property
    def outputs(self):
        return {"time_task1": File_Txt(self.node.outputs[0])}

    def run(self):
        DS_DATA_HOME = self.inputs["DS_DATA_HOME"].read_loc()
        start = time.time()
        print("start time TASK1: ", start)
        for i in range(5):
            # sleep for 5 seconds
            time.sleep(5)
        stop = time.time()
        print("stop time TASK1: ", stop)
        self.outputs["time_task1"].write(str(stop))

class Task2(BaseOperator):

    @property
    def inputs(self):
        return {"time_task1": File_Txt(
                self.node.inputs[0])
                }

    @property
    def outputs(self):
        return {"time_task2": File_Txt(
                self.node.outputs[0])}

    def run(self):
        time_task1 = self.inputs["time_task1"].read()
        print("task1 stopped at:", time_task1)
        start = time.time()
        print("start time TASK2: ", start)
        for i in range(5):
            # sleep for 5 seconds
            time.sleep(5)
        stop = time.time()
        print("stop time TASK2: ", stop)
        self.outputs["time_task2"].write(str(stop))

class Task3(BaseOperator):
    
    @property
    def inputs(self):
        return {"time_task2": File_Txt(
                self.node.inputs[0])
                }

    @property
    def outputs(self):
        return {"time_task3": File_Txt(
                self.node.outputs[0])}

    def run(self):
        time_task2 = self.inputs["time_task2"].read()
        print("task2 stopped at:", time_task2)
        start = time.time()
        print("start time TASK3: ", start)
        for i in range(5):
            # sleep for 5 seconds
            time.sleep(5)
        stop = time.time()
        print("stop time TASK3: ", stop)
        self.outputs["time_task3"].write(str(stop))
