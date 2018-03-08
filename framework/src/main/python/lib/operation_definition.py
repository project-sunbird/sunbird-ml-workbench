from pandas import HDFStore
from utils.common import dir_loc, create_directory, get_parent_dir
import numpy as np


##
# Node Operation
##

class NodeOperation(object):

    def __init__(self, node):

        self.node_name = node.node_name
        self.inputs = node.inputs
        self.outputs = node.outputs
        self.arguments = node.arguments
        self.experiment_dir = dir_loc(node.experiment_dir)
        self.graph_inputs = node.graph_inputs
        self.graph_outputs = node.graph_outputs
        self._output_node_mapping = np.load(self.experiment_dir+'output_node_mapping.npy').item()

        create_directory(dir_loc(self.experiment_dir + self.node_name))

        if self.arguments is None:
            self.arguments = {}

    def get_location(self, reference, type):

        # Create location for reference based on type
        # from graph inputs
        if reference in self.graph_inputs:
            location = self.graph_inputs[reference]
        # from graph outputs
        elif reference in self.graph_outputs:
            location = self.graph_outputs[reference]
            create_directory(get_parent_dir(location))
            # graph output create directory must be handled in the node
        # internal io object
        else:
            if type == 'file':
                input_node = self._output_node_mapping[reference]
                location = dir_loc(self.experiment_dir + input_node) + reference
                create_directory(get_parent_dir(location))
            elif type == 'folder':
                input_node = self._output_node_mapping[reference]
                location = dir_loc(self.experiment_dir + input_node + '/' + reference)
                create_directory(location)
            elif type == 'dataframe':
                input_node = self._output_node_mapping[reference]
                location = dir_loc(self.experiment_dir + input_node) + input_node + '.h5'
                create_directory(get_parent_dir(location))
            else:
                raise ValueError("type should be 'file', 'folder' or 'dataframe'")

        # Might not be required - Can make hdf5 storage unstable!
        # io_details = pd.DataFrame([{'Name':reference, 'Type': type, 'Location': location}])
        # self.store.open()
        # self.store.append('io_details',io_details, min_itemsize = { 'Location' : 200 }) # multiple entries are possible
        # self.store.close()

        return location

    def io_check(self, condition):

        if not(condition):
            raise ValueError("{} node has unexpected number of I/O objects.".format(self.node_name))

    def get_dataframes(self, list_of_references):
        # takes a list of references and returns a list of dataframes
        df_list = []

        for i in range(0, len(list_of_references)):

            input_name = list_of_references[i]
            input_node = self._output_node_mapping[input_name]
            input_store = HDFStore(dir_loc(self.experiment_dir + input_node) + input_node + '.h5')

            data = input_store.get(input_name)
            df_list.append(data)

        return df_list

    def put_dataframes(self, df_list, references):
        # takes a list of dataframes and stores it in HDFStore
        dataframe_store = HDFStore(dir_loc(self.experiment_dir + self.node_name) + self.node_name + '.h5')
        if len(df_list) == len(references):
            dataframe_store.open()
            for i in range(0,len(df_list)):
                dataframe_store.put(references[i], df_list[i], 't')
            dataframe_store.close()
        else:
            raise ValueError("{} node has mismatch between output objects generated and assigned to the node."
                             .format(self.node_name))

    # Reader - helper
    def read_files_to_dataframe_storage(self, func):
        # takes a reader function and reads files to HDFStore

        for i in range(0,len(self.inputs)):
            input_file_loc = self.graph_inputs[self.inputs[i]]
            data = func(input_file_loc, **self.arguments)

            dataframe_store = HDFStore(dir_loc(self.experiment_dir + self.node_name) + self.node_name + '.h5')
            dataframe_store.put(self.outputs[i], data, 't')
            dataframe_store.close()

    # Writer - helper
    def get_dataframe_objects_to_write(self):
        # returns list of dataframes and list of locations to write files

        df_list = []
        output_loc_list = []
        for i in range(0,len(self.outputs)):

            input_name = self.inputs[i]
            input_node = self._output_node_mapping[input_name]
            input_store = HDFStore(dir_loc(self.experiment_dir + input_node) + input_node + '.h5')

            data = input_store.get(input_name)
            df_list.append(data)

            # create output directory
            output_file_loc = self.graph_outputs[self.outputs[i]]
            create_directory(get_parent_dir(output_file_loc))

            output_loc_list.append(output_file_loc)

        return df_list, output_loc_list
