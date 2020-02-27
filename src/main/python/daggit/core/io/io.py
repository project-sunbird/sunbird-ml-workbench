from future.builtins import super
from abc import abstractmethod
import warnings
import pandas as pd
import pickle
import os
import json
import logging
import configparser

from ..base.config import DAGGIT_HOME, STORE, STORAGE_FORMAT
from ..base.utils import create_dir
from ..base.config import STORAGE_FORMAT

from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from kafka import KafkaClient
from kafka import KafkaProducer, KafkaConsumer

# TODO configuration file for settings


class DataType(object):

    def __init__(self, data_pointer, **kwargs):

        if data_pointer.external_ref is None:
            self.data_location = self.get_temp_path(
                data_pointer.dag_id,
                data_pointer.parent_task,
                data_pointer.data_alias + STORAGE_FORMAT)
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
            store_path = os.path.join(
                os.getenv("DAGGIT_HOME"), "daggit_storage")

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
                warnings.warn(
                    'Reading the input file as a tab/space delimited file. \n' +
                    'Please verify your input or use csv, HDF5 or josn format')
                return pd.read_csv(self.data_location, sep=" ")
            except pd.errors.ParserError:
                raise IOError(
                    'Unidentified Data format. Please use csv, HDF5 or json files for input')

    def write(self, data):
        if self.data_location[-3:] == 'csv':
            create_dir(os.path.dirname(self.data_location))
            data.to_csv(path_or_buf=self.data_location, index=False)
        elif self.data_location[-2:] == 'h5':
            create_dir(os.path.dirname(self.data_location))
            dataframe_store = pd.HDFStore(self.data_location)
            dataframe_store.put(
                key=self.data_alias,
                value=data,
                format='t',
                append=True)
            dataframe_store.close()
        elif self.data_location[-4:] == 'json':
            create_dir(os.path.dirname(self.data_location))
            data.to_json(path_or_buf=self.data_location)
        else:

            warnings.warn(
                'Writing the dataframe into a tab/space delimited file. \n' +
                'Please verify your output file or use csv, HDF5 or josn format')
            data.to_csv(
                self.data_location +
                '.txt',
                header=True,
                index=False,
                sep=' ',
                mode='a')


class CSV_Pandas(DataType): #provides interface for the operators 

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
        dataframe_store.put(
            key=self.data_alias,
            value=data,
            format='t',
            append=True)
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
        print(
            "dirname for data_location:",
            os.path.dirname(
                self.data_location))
        create_dir(os.path.dirname(self.data_location))
        f = open(self.data_location, "w+")
        f.write(data)
        f.close()


class File_IO(DataType):
    
    def read(self):
        if self.data_location[-3:] == STORAGE_FORMAT:
            with open(self.data_location, "r") as file:
                return file.read()
        else:
            return self.data_location

    def write(self, data):
        print(
            "dirname for data_location:",
            os.path.dirname(
                self.data_location))
        create_dir(os.path.dirname(self.data_location))
        with open(self.data_location, "w+") as write_file:
            write_file.write(data)


class File_JSON(DataType):

    def location_specify(self):
        return self.data_location

    def read(self):
        f = open(self.data_location, "r")
        return f.read()

    def write(self, data):
        print(
            "dirname for data_location:",
            os.path.dirname(
                self.data_location))
        create_dir(os.path.dirname(self.data_location))
        with open(self.data_location, 'w+') as json_file:
            json.dump(data, json_file, indent=4)


class Pickle_Obj(DataType):

    def read(self):
        with open(self.data_location, 'rb') as handle:
            data = pickle.load(handle)
        return data

    def write(self, data):
        create_dir(os.path.dirname(self.data_location))
        with open(self.data_location, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

class KafkaDispatcher(DataType):
    
    def __init__(self, configfile):
        self.conf = configfile
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(self.conf)
        self.host = config["kafka"]["host"]
        self.port = config["kafka"]["port"]
        self.kafka_broker = self.host + ":" + self.port
        self.connection_established = False
        try:
            self.client = KafkaClient(self.kafka_broker)
            self.server_topics = self.client.topic_partitions
            self.connection_established = True
        except Exception as e:
            print(e)
            pass
        
    @abstractmethod
    def read(self):
        raise NotImplementedError()

    def write(self, transaction_event, writeTotopic):
        
        self.write_to_topic = writeTotopic
        self.event_json = transaction_event
        print("topic: {0}".format(self.write_to_topic))
        print("host: {0}".format(self.host))
        print("port: {0}".format(self.port))
        print("kafka_broker: {0}".format(self.kafka_broker))
        print("Connection established: {0}".format(self.connection_established))
        print("Server topics {0}".format(self.server_topics))
        
         # check if connection is established.
        flag = True
        if self.connection_established:
            if self.write_to_topic in self.server_topics.keys():
                try:
                    producer = KafkaProducer(bootstrap_servers=self.kafka_broker, value_serializer=lambda v: json.dumps(v, indent=4).encode('utf-8'))
                    # serializing json message:-
                    event_send = producer.send(self.write_to_topic, transaction_event)
                    result = event_send.get(timeout=60)
                    flag = True
                except Exception as e:
                    print(e)
                    flag = False
            else:
                print("Topic doesnot exist!!")
                flag = False
        else:
            print("Connection to KafkaServer is not established")
            flag = False
        return flag
        
class KafkaCLI(KafkaDispatcher):
    
    def __init__(self, configfile):
        KafkaDispatcher.__init__(self, configfile)
        self.append_event = []
    
    def write(self, transaction_event, writeTotopic):
        flag_status = KafkaDispatcher.write(self, transaction_event, writeTotopic)
        return flag_status

    def read(self, read_from_topic, groupID, offset_reset, session_timeout, auto_commit_enable):
        self.read_from_topic = read_from_topic
        self.groupID = groupID
        self.offset_reset = offset_reset
        self.session_timeout = session_timeout
        self.auto_commit_enable = auto_commit_enable
        if self.connection_established:
            if self.read_from_topic in self.server_topics.keys():
                consumer = KafkaConsumer(self.read_from_topic, 
                                         bootstrap_servers= self.kafka_broker,
                                         value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                                         group_id=self.groupID,      
                                         enable_auto_commit= auto_commit_enable,
                                         session_timeout_ms = self.session_timeout,
                                         auto_offset_reset = self.offset_reset)
                try:
                    while True:
                        for i, message in enumerate(consumer):
                            event = message.value
                            self.append_event.append(event)
                            print(event)
                except KeyboardInterrupt:
                    pass
                finally:
                    consumer.close()
            else:
                print("Topic does not exist!")
        else:
            print("Connection to KafkaServer is not established")
        return self.append_event   
    
class KafkaStreaming(KafkaDispatcher):
    def __init__(self, configfile):
        KafkaDispatcher.__init__(self, configfile)
        self.append_event = []
    
    def write(self, transaction_event, writeTotopic):
        flag_status = KafkaDispatcher.write(self, transaction_event, writeTotopic)
        return flag_status

    def handler(self, message):
        records = self.message.collect()
        for record in records:
            self.append_event.append(record)

    def read(self, app_name, read_from_topic, groupID, offset_reset, session_timeout, auto_commit_enable, batch_duration):
        self.read_from_topic = read_from_topic
        self.groupID = groupID
        self.app_name = app_name
        self.offset_reset = offset_reset
        self.session_timeout = session_timeout
        self.auto_commit_enable = auto_commit_enable
        self.batch_duration = batch_duration
        
        self.kafkaParams = {"metadata.broker.list": self.kafka_broker}
        self.kafkaParams["auto.offset.reset"] = self.offset_reset
        self.kafkaParams["enable.auto.commit"] = self.auto_commit_enable
        
        if self.connection_established:
            if self.write_to_topic in self.server_topics.keys():
                sc = SparkContext(app_name=self.app_name)
                ssc = StreamingContext(sc, self.batch_duration)

                kvs = KafkaUtils.createDirectStream(ssc, [self.read_from_topic], self.kafkaParams)#{"metadata.broker.list": brokers})
                kvs.foreachRDD(handler)

                ssc.start()
                ssc.awaitTermination()