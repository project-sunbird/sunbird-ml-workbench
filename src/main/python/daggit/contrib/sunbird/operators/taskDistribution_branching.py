import time
import os

from daggit.core.io.io import Pandas_Dataframe, File_Txt
from daggit.core.io.io import ReadDaggitTask_Folderpath
from daggit.core.base.factory import BaseOperator


class Content1(BaseOperator):

    @property
    def inputs(self):
        return {"DS_DATA_HOME": ReadDaggitTask_Folderpath(
                self.node.inputs[0])
                }

    @property
    def outputs(self):
        return {"time_content1": File_Txt(self.node.outputs[0])}

    def run(self):
        DS_DATA_HOME = self.inputs["DS_DATA_HOME"].read_loc()
        start = time.time()
        print("start time for content1: ", start)
        stop = time.time()
        print("stop time for content1: ", stop)
        self.outputs["time_content1"].write(str(stop))

class Content2(BaseOperator):

    @property
    def inputs(self):
        return {"DS_DATA_HOME": ReadDaggitTask_Folderpath(
                self.node.inputs[0])
                }

    @property
    def outputs(self):
        return {"time_content2": File_Txt(self.node.outputs[0])}

    def run(self):
        DS_DATA_HOME = self.inputs["DS_DATA_HOME"].read_loc()
        start = time.time()
        print("start time for content2: ", start)
        stop = time.time()
        print("stop time for content2: ", stop)
        self.outputs["time_content2"].write(str(stop))

class Content3(BaseOperator):

    @property
    def inputs(self):
        return {"DS_DATA_HOME": ReadDaggitTask_Folderpath(
                self.node.inputs[0])
                }

    @property
    def outputs(self):
        return {"time_content3": File_Txt(self.node.outputs[0])}

    def run(self):
        DS_DATA_HOME = self.inputs["DS_DATA_HOME"].read_loc()
        start = time.time()
        print("start time for content3: ", start)
        stop = time.time()
        print("stop time for content3: ", stop)
        self.outputs["time_content3"].write(str(stop))

class Content4(BaseOperator):

    @property
    def inputs(self):
        return {"DS_DATA_HOME": ReadDaggitTask_Folderpath(
                self.node.inputs[0])
                }

    @property
    def outputs(self):
        return {"time_content4": File_Txt(self.node.outputs[0])}

    def run(self):
        DS_DATA_HOME = self.inputs["DS_DATA_HOME"].read_loc()
        start = time.time()
        print("start time for content4: ", start)
        stop = time.time()
        print("stop time for content4: ", stop)
        self.outputs["time_content4"].write(str(stop))

class Content5(BaseOperator):

    @property
    def inputs(self):
        return {"DS_DATA_HOME": ReadDaggitTask_Folderpath(
                self.node.inputs[0])
                }

    @property
    def outputs(self):
        return {"time_content5": File_Txt(self.node.outputs[0])}

    def run(self):
        DS_DATA_HOME = self.inputs["DS_DATA_HOME"].read_loc()
        start = time.time()
        print("start time for content5: ", start)
        stop = time.time()
        print("stop time for content5: ", stop)
        self.outputs["time_content5"].write(str(stop))

class C1_Task1(BaseOperator):

    @property
    def inputs(self):
        return {"time_content1": File_Txt(
                self.node.inputs[0])}

    @property
    def outputs(self):
        return {"c1_time_task1": File_Txt(
                self.node.outputs[0])}

    def run(self):
        time_content1 = self.inputs["time_content1"].read()
        start = time.time()
        print("start time TASK1: ", start)
        # sleep for 5 seconds
        time.sleep(5)
        stop = time.time()
        diff = stop-start
        print("time difference:", diff)
        print("stop time TASK1: ", stop)
        self.outputs["c1_time_task1"].write(str(stop))

class C1_Task2(BaseOperator):
    
    @property
    def inputs(self):
        return {"time_content1": File_Txt(
                self.node.inputs[0])}

    @property
    def outputs(self):
        return {"c1_time_task2": File_Txt(
                self.node.outputs[0])}

    def run(self):
        time_content1 = self.inputs["time_content1"].read()
        start = time.time()
        print("start time TASK2: ", start)
        # sleep for 5 seconds
        time.sleep(5)
        stop = time.time()
        diff = stop-start
        print("time difference:", diff)
        print("stop time TASK2: ", stop)
        self.outputs["c1_time_task2"].write(str(stop))

class C1_Task3(BaseOperator):
    
    @property
    def inputs(self):
        return {"time_content1": File_Txt(
                self.node.inputs[0])}

    @property
    def outputs(self):
        return {"c1_time_task3": File_Txt(
                self.node.outputs[0])}

    def run(self):
        time_content1 = self.inputs["time_content1"].read()
        start = time.time()
        print("start time TASK3: ", start)
        # sleep for 5 seconds
        time.sleep(5)
        stop = time.time()
        diff = stop-start
        print("time difference:", diff)
        print("stop time TASK3: ", stop)
        self.outputs["c1_time_task3"].write(str(stop))

class C2_Task1(BaseOperator):
    
    @property
    def inputs(self):
        return {"time_content2": File_Txt(
                self.node.inputs[0])}

    @property
    def outputs(self):
        return {"c2_time_task1": File_Txt(
                self.node.outputs[0])}

    def run(self):
        time_content2 = self.inputs["time_content2"].read()
        start = time.time()
        print("start time TASK1: ", start)
        # sleep for 5 seconds
        time.sleep(5)
        stop = time.time()
        diff = stop-start
        print("time difference:", diff)
        print("stop time TASK1: ", stop)
        self.outputs["c2_time_task1"].write(str(stop))

class C2_Task2(BaseOperator):
    
    @property
    def inputs(self):
        return {"time_content2": File_Txt(
                self.node.inputs[0])}

    @property
    def outputs(self):
        return {"c2_time_task2": File_Txt(
                self.node.outputs[0])}

    def run(self):
        time_content2 = self.inputs["time_content2"].read()
        start = time.time()
        print("start time TASK2: ", start)
        # sleep for 5 seconds
        time.sleep(5)
        stop = time.time()
        diff = stop-start
        print("time difference:", diff)
        print("stop time TASK2: ", stop)
        self.outputs["c2_time_task2"].write(str(stop))

class C2_Task3(BaseOperator):
    
    @property
    def inputs(self):
        return {"time_content2": File_Txt(
                self.node.inputs[0])}

    @property
    def outputs(self):
        return {"c2_time_task3": File_Txt(
                self.node.outputs[0])}

    def run(self):
        time_content2 = self.inputs["time_content2"].read()
        start = time.time()
        print("start time TASK3: ", start)
        # sleep for 5 seconds
        time.sleep(5)
        stop = time.time()
        diff = stop-start
        print("time difference:", diff)
        print("stop time TASK3: ", stop)
        self.outputs["c2_time_task3"].write(str(stop))

class C3_Task1(BaseOperator):
    
    @property
    def inputs(self):
        return {"time_content3": File_Txt(
                self.node.inputs[0])}

    @property
    def outputs(self):
        return {"c3_time_task1": File_Txt(
                self.node.outputs[0])}

    def run(self):
        time_content3 = self.inputs["time_content3"].read()
        start = time.time()
        print("start time TASK1: ", start)
        # sleep for 5 seconds
        time.sleep(5)
        stop = time.time()
        diff = stop-start
        print("time difference:", diff)
        print("stop time TASK1: ", stop)
        self.outputs["c3_time_task1"].write(str(stop))

class C3_Task2(BaseOperator):
    
    @property
    def inputs(self):
        return {"time_content3": File_Txt(
                self.node.inputs[0])}

    @property
    def outputs(self):
        return {"c3_time_task2": File_Txt(
                self.node.outputs[0])}

    def run(self):
        time_content3 = self.inputs["time_content3"].read()
        start = time.time()
        print("start time TASK2: ", start)
        # sleep for 5 seconds
        time.sleep(5)
        stop = time.time()
        diff = stop-start
        print("time difference:", diff)
        print("stop time TASK2: ", stop)
        self.outputs["c3_time_task2"].write(str(stop))

class C3_Task3(BaseOperator):
    
    @property
    def inputs(self):
        return {"time_content3": File_Txt(
                self.node.inputs[0])}

    @property
    def outputs(self):
        return {"c3_time_task3": File_Txt(
                self.node.outputs[0])}

    def run(self):
        time_content3 = self.inputs["time_content3"].read()
        start = time.time()
        print("start time TASK3: ", start)
        # sleep for 5 seconds
        time.sleep(5)
        stop = time.time()
        diff = stop-start
        print("time difference:", diff)
        print("stop time TASK3: ", stop)
        self.outputs["c3_time_task3"].write(str(stop))

class C4_Task1(BaseOperator):
    
    @property
    def inputs(self):
        return {"time_content4": File_Txt(
                self.node.inputs[0])}

    @property
    def outputs(self):
        return {"c4_time_task1": File_Txt(
                self.node.outputs[0])}

    def run(self):
        time_content4 = self.inputs["time_content4"].read()
        start = time.time()
        print("start time TASK1: ", start)
        # sleep for 5 seconds
        time.sleep(5)
        stop = time.time()
        diff = stop-start
        print("time difference:", diff)
        print("stop time TASK1: ", stop)
        self.outputs["c4_time_task1"].write(str(stop))

class C4_Task2(BaseOperator):
    
    @property
    def inputs(self):
        return {"time_content4": File_Txt(
                self.node.inputs[0])}

    @property
    def outputs(self):
        return {"c4_time_task2": File_Txt(
                self.node.outputs[0])}

    def run(self):
        time_content4 = self.inputs["time_content4"].read()
        start = time.time()
        print("start time TASK2: ", start)
        # sleep for 5 seconds
        time.sleep(5)
        stop = time.time()
        diff = stop-start
        print("time difference:", diff)
        print("stop time TASK2: ", stop)
        self.outputs["c4_time_task2"].write(str(stop))

class C4_Task3(BaseOperator):
    
    @property
    def inputs(self):
        return {"time_content4": File_Txt(
                self.node.inputs[0])}

    @property
    def outputs(self):
        return {"c4_time_task3": File_Txt(
                self.node.outputs[0])}

    def run(self):
        time_content4 = self.inputs["time_content4"].read()
        start = time.time()
        print("start time TASK3: ", start)
        # sleep for 5 seconds
        time.sleep(5)
        stop = time.time()
        diff = stop-start
        print("time difference:", diff)
        print("stop time TASK3: ", stop)
        self.outputs["c4_time_task3"].write(str(stop))

class C5_Task1(BaseOperator):
    
    @property
    def inputs(self):
        return {"time_content5": File_Txt(
                self.node.inputs[0])}

    @property
    def outputs(self):
        return {"c5_time_task1": File_Txt(
                self.node.outputs[0])}

    def run(self):
        time_content5 = self.inputs["time_content5"].read()
        start = time.time()
        print("start time TASK1: ", start)
        # sleep for 5 seconds
        time.sleep(5)
        stop = time.time()
        diff = stop-start
        print("time difference:", diff)
        print("stop time TASK1: ", stop)
        self.outputs["c5_time_task1"].write(str(stop))

class C5_Task2(BaseOperator):
    
    @property
    def inputs(self):
        return {"time_content5": File_Txt(
                self.node.inputs[0])}

    @property
    def outputs(self):
        return {"c5_time_task2": File_Txt(
                self.node.outputs[0])}

    def run(self):
        time_content5 = self.inputs["time_content5"].read()
        start = time.time()
        print("start time TASK2: ", start)
        # sleep for 5 seconds
        time.sleep(5)
        stop = time.time()
        diff = stop-start
        print("time difference:", diff)
        print("stop time TASK2: ", stop)
        self.outputs["c5_time_task2"].write(str(stop))

class C5_Task3(BaseOperator):
    
    @property
    def inputs(self):
        return {"time_content5": File_Txt(
                self.node.inputs[0])}

    @property
    def outputs(self):
        return {"c5_time_task3": File_Txt(
                self.node.outputs[0])}

    def run(self):
        time_content5 = self.inputs["time_content5"].read()
        start = time.time()
        print("start time TASK3: ", start)
        # sleep for 5 seconds
        time.sleep(5)
        stop = time.time()
        diff = stop-start
        print("time difference:", diff)
        print("stop time TASK3: ", stop)
        self.outputs["c5_time_task3"].write(str(stop))

