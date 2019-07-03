import os
import glob
import json
import requests
import logging
import shutil
import nltk
import time
import configparser
import pandas as pd
import numpy as np

from daggit.core.base.factory import BaseOperator
from daggit.core.io.io import Pandas_Dataframe, File_Txt
from daggit.core.io.io import ReadDaggitTask_Folderpath
from daggit.contrib.sunbird.oplib.taggingUtils import multimodal_text_enrichment
from daggit.core.oplib.misc import df_feature_check, identify_contentType
from daggit.contrib.sunbird.oplib.profanityUtils import uncommonWords
from daggit.contrib.sunbird.oplib.profanityUtils import betterProfanity
from daggit.contrib.sunbird.oplib.profanityUtils import profanityFilter
from daggit.contrib.sunbird.oplib.profanityUtils import docProfanity
from daggit.contrib.sunbird.oplib.profanityUtils import profanityCheck
from daggit.contrib.sunbird.oplib.profanityUtils import text_profanity

class ContentToTextRead(BaseOperator):
    """
    An operator that extracts text for Content id within a specified range, in a Content meta dataframe

    :param range_start: specifies the start index of the dataframe 
    :type range_start: int
    :param range_end: specifies the end index of the dataframe
    :type range_end: int
    :param num_of_processes: number of workers used to distribute the process
    :type num_of_processes: int
    :param content_type: Specifies  the type of the content and related parameters
        Options are: ``{ youtube | pdf | ecml }
    :type content_type: json
    """
    @property
    def inputs(self):
        """
        Function that the ContentToTextRead operator defines while returning graph inputs

        :returns: Inputs to the node of the Auto tagging graph
            DS_DATA_HOME: a localpath where the folders get created
            localpathTocontentMeta: path to content meta
            pathTocredentials: path to config file with credentials 

        """
        return {
                "DS_DATA_HOME": ReadDaggitTask_Folderpath(self.node.inputs[0]),
                "localpathTocontentMeta": ReadDaggitTask_Folderpath(self.node.inputs[1]),
                "pathTocredentials": ReadDaggitTask_Folderpath(self.node.inputs[2])
                }

    @property
    def outputs(self):
        """
        Function that the ContentToTextRead operator defines while returning graph outputs

        :returns: Returns the path to timestamp folder in which auto tagging results get generated

        """
        return {"timestamp_folder": File_Txt(
                self.node.outputs[0])}

    def run(
            self,
            range_start,
            range_end,
            num_of_processes,
            content_type):
        """
        This is the main method to derive when creating an operator. This takes in the parameters, 
        runs text enrichment pipline and writes back the path to the 
        timestamp folder with the content id and its enriched text to an h5 file that gets saved as an intermediate result  

        """
        DS_DATA_HOME = self.inputs["DS_DATA_HOME"].read_loc()
        pathTocredentials = self.inputs["pathTocredentials"].read_loc()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        path_to_timestamp_folder = os.path.join(DS_DATA_HOME, timestr)
        content_to_text_path = os.path.join(
            path_to_timestamp_folder, "content_to_text")
        # content dump:
        if not os.path.exists(content_to_text_path):
            os.makedirs(content_to_text_path)
            print("content_to_text: ", content_to_text_path)
        
        contentmeta_path = self.inputs["localpathTocontentMeta"].read_loc()
        # move the content meta to timestamp folder[destination folder]
        #for the time being experiment with copy: change it later.
        shutil.copy(contentmeta_path, os.path.join(path_to_timestamp_folder, os.path.split(contentmeta_path)[1]))
        moved_contentmeta_path = os.path.join(path_to_timestamp_folder, os.path.split(contentmeta_path)[1])
        
        content_meta = pd.read_csv(moved_contentmeta_path)
        if "derived_contentType" not in list(content_meta.columns):
            content_meta["derived_contentType"] = np.nan
            for row_ind, artifact_url in enumerate(content_meta["artifactUrl"]):
                try:
                    content_meta["derived_contentType"][row_ind] = identify_contentType(artifact_url)
                except BaseException:
                    pass
        content_meta = content_meta[pd.notnull(content_meta['derived_contentType'])]
        content_meta.reset_index(inplace=True, drop=True)
        print(self.outputs["timestamp_folder"].location_specify())
        oldwd = os.getcwd()
        contentMeta_mandatory_fields = [
            'artifactUrl',
            'derived_contentType',
            'downloadUrl',
            'gradeLevel',
            'identifier',
            'language',
            'subject',
            'graph_id',
            'nodeType',
            'objectType',
            'node_id']
        # assert df_feature_check(content_meta, contentMeta_mandatory_fields)

        logging.info("CTT_CONTENT_TO_TEXT_START")
        # read content meta:
        if content_meta.columns[0] == "0":
            content_meta = content_meta.drop("0", axis=1)

        # check for duplicates in the meta
        if list(content_meta[content_meta.duplicated(
                ['artifactUrl'], keep=False)]["artifactUrl"]) != []:
            content_meta.drop_duplicates(subset="artifactUrl", inplace=True)
            content_meta.reset_index(drop=True, inplace=True)

        # dropna from artifactUrl feature and reset the index:
        content_meta.dropna(subset=["artifactUrl"], inplace=True)
        content_meta.reset_index(drop=True, inplace=True)

        # time the run
        start = time.time()
        logging.info(
            'Contents detected in the content meta: ' + str(len(content_meta)))
        logging.info(
            "----Running Content_to_Text for contents from {0} to {1}:".format(
                range_start, range_end))
        logging.info("time started: {0}".format(start))
        # subset contentMeta:
        # content_meta = content_meta[content_meta["derived_contentType"].isin(
        #     subset_contentMeta_by.split(", "))]
        content_meta.reset_index(drop=True, inplace=True)
        if range_start == "START":
            range_start = 0
        if range_end == "END":
            range_end = len(content_meta)
        logging.info(
            "CTT_Config: content_meta from {0} to {1} created in: {2}".format(
                range_start, range_end, content_to_text_path))
        print("Number of processes: ", num_of_processes)

        status = False
        if os.path.exists(pathTocredentials):
            try:
                config = configparser.ConfigParser(allow_no_value=True)
                config.read(pathTocredentials)
                status = True
                try:
                    path_to_googlecred = config['google application credentials']["GOOGLE_APPLICATION_CREDENTIALS"]
                    with open(path_to_googlecred, "r") as cred_json:
                        GOOGLE_APPLICATION_CREDENTIALS = cred_json.read()
                except BaseException:
                    logging.info("Invalid GOOGLE_APPLICATION_CREDENTIALS in config.")
                    logging.info("***Checking for GOOGLE_APPLICATION_CREDENTIALS environment variable")
                    status = False
            except BaseException:
                logging.info("Invalid config file")
                logging.info("***Checking for GOOGLE_APPLICATION_CREDENTIALS environment variable")

        if not status:
            try:
                GOOGLE_APPLICATION_CREDENTIALS = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
                with open(GOOGLE_APPLICATION_CREDENTIALS, "r") as f:
                    GOOGLE_APPLICATION_CREDENTIALS = f.read()
            except BaseException:
                GOOGLE_APPLICATION_CREDENTIALS = ""
                logging.info("Not a valid google credential") 

        result = [
            multimodal_text_enrichment(
                i,
                timestr,
                content_meta,
                content_type,
                content_to_text_path,
                GOOGLE_APPLICATION_CREDENTIALS) for i in range(
                range_start,
                range_end)]
        print(result)
        os.chdir(oldwd)
        print("Current directory c2t: ", os.getcwd())
        print("timestamp_folder path:", path_to_timestamp_folder)
        self.outputs["timestamp_folder"].write(path_to_timestamp_folder)


class ProfanityCheck(BaseOperator):
    """ 
    An operator that returns profane words from a given text

    :param cid_start: specifies the start index of cid in the folder
    :type cid_start: int
    :param cid_end: specifies the end index of cid in the folder
    :type cid_end: int
    """
    @property
    def inputs(self):
        return {
                "DS_DATA_HOME": ReadDaggitTask_Folderpath(self.node.inputs[0]),
                "timestamp_folder": File_Txt(self.node.inputs[1])}


    def run(self):
        DS_DATA_HOME = self.inputs["DS_DATA_HOME"].read_loc()
        timestamp_folder = self.inputs["timestamp_folder"].read()
        read_basepath = os.path.join(DS_DATA_HOME, timestamp_folder, "content_to_text")
        write_basepath = os.path.join(DS_DATA_HOME, timestamp_folder, "content_review")
        cids_list = os.listdir(read_basepath)
        cids_list.remove('.DS_Store')
        for cid in cids_list:
            try:
                content_info_file=os.path.join(read_basepath,cid,"ML_content_info.json")
                with open(content_info_file) as json_file:
                    data = json.load(json_file)
                text = data['text']
                profanity_filter_op = text_profanity(text)
                profanity_filter_op.update({"identifier": cid})
                with open(os.path.join(write_basepath, cid + '.json'), 'w') as outfile:
                    json.dump(profanity_filter_op, outfile)
            except:
                pass
        print("Path to content review: ", read_basepath)