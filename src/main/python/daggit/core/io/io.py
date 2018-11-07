from future.builtins import super
from abc import abstractmethod
import warnings
import pandas as pd
import pickle
import os
from ..base.config import DAGGIT_HOME, STORE
from ..base.utils import create_dir
from ..base.config import STORAGE_FORMAT

# TODO configuration file for settings

class DataType(object):

    def __init__(self, data_pointer, **kwargs):

        if data_pointer.external_ref is None:
            self.data_location = self.get_temp_path(data_pointer.dag_id,
                                                    data_pointer.parent_task, data_pointer.data_alias+STORAGE_FORMAT)
        else:
            self.data_location = data_pointer.external_ref

        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def read(self):
        raise NotImplementedError()

    @abstractmethod
    def write(self, data):
        raise NotImplementedError()

    @staticmethod
    def get_temp_path(dag_id, task_id=None, data_alias=None):

        os.environ['DAGGIT_HOME'] = DAGGIT_HOME
        store_type = STORE

        if store_type == 'Local':
            store_path = os.path.join(os.getenv("DAGGIT_HOME"), "daggit_storage")
             
        else:
            raise ValueError("Unrecognised store type.")

        if task_id is None:
            path = os.path.join(store_path, dag_id)
        else:
            if data_alias is None:
                path = os.path.join(store_path, dag_id, task_id)
            else:
                path = os.path.join(store_path, dag_id, task_id, data_alias)
        
        return path


class Pandas_Dataframe(DataType):

    def __init__(self, data_pointer, **kwargs):
        super().__init__(data_pointer, **kwargs)
        self.data_alias = data_pointer.data_alias

    def read(self):
        print("data_location: ", self.data_location)
        if self.data_location[-3:] == 'csv':
            return pd.read_csv(filepath_or_buffer=self.data_location)
        elif self.data_location[-2:] == 'h5':
            store = pd.HDFStore(self.data_location)
            return store.get(self.data_alias)
        elif self.data_location[-4:] == 'json':
            return pd.read_json(path_or_buf=self.data_location)
        else:
            try:
                warnings.warn('Reading the input file as a tab/space delimited file. \n'+
                              'Please verify your input or use csv, HDF5 or josn format')
                return pd.read_csv(self.data_location, sep = " ")
            except pd.errors.ParserError:
                raise IOError('Unidentified Data format. Please use csv, HDF5 or json files for input')


    def write(self, data):
        if self.data_location[-3:] == 'csv':
            create_dir(os.path.dirname(self.data_location))
            data.to_csv(path_or_buf=self.data_location, index=False)
        elif self.data_location[-2:] == 'h5':
            create_dir(os.path.dirname(self.data_location))
            dataframe_store = pd.HDFStore(self.data_location)
            dataframe_store.put(key=self.data_alias, value=data, format='t', append=True)
            dataframe_store.close()
        elif self.data_location[-4:] == 'json':
            create_dir(os.path.dirname(self.data_location))
            data.to_json(path_or_buf=self.data_location)
        else:

            warnings.warn('Writing the dataframe into a tab/space delimited file. \n' +
                          'Please verify your output file or use csv, HDF5 or josn format')
            data.to_csv(self.data_location+'.txt', header=True, index=False, sep=' ', mode='a')


class CSV_Pandas(DataType):

    def read(self):
        return pd.read_csv(filepath_or_buffer=self.data_location)

    def write(self, data):
        create_dir(os.path.dirname(self.data_location))
        data.to_csv(path_or_buf=self.data_location, index=False)


class HDF_Pandas(DataType):

    def __init__(self, data_pointer, **kwargs):
        super().__init__(data_pointer, **kwargs)
        self.data_alias = data_pointer.data_alias

    def read(self):
        store = pd.HDFStore(self.data_location)
        return store.get(self.data_alias)

    def write(self, data):
        create_dir(os.path.dirname(self.data_location))
        dataframe_store = pd.HDFStore(self.data_location)
        dataframe_store.put(key=self.data_alias, value=data, format='t', append=True)
        dataframe_store.close()

class ReadDaggitTask_Folderpath(DataType): 

    def read_loc(self):
        return self.data_location


class File_Txt(DataType):
    
    def location_specify(self):
        return self.data_location

    def read(self):
        f = open(self.data_location, "r")
        return f.read()

    def write(self, data):
        print("dirname for data_location:", os.path.dirname(self.data_location))
        create_dir(os.path.dirname(self.data_location))
        f = open(self.data_location, "w+")
        f.write(data)
        f.close()


class Pickle_Obj(DataType):

    def read(self):
        with open(self.data_location, 'rb') as handle:
            data = pickle.load(handle)
        return data

    def write(self, data):
        create_dir(os.path.dirname(self.data_location))
        with open(self.data_location, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
