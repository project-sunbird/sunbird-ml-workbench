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
from daggit.contrib.sunbird.nodes.taggingNodes import ContentToTextRead


class ProfanityCheck(BaseOperator):
    """ 
    An operator that returns profane words from a given text
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
        if '.DS_Store' in cids_list:
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