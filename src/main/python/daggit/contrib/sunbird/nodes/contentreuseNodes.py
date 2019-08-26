import sys
import os
import daggit
import requests
import io
import time
import re
import logging
import configparser

from google.cloud import vision
from google.cloud import storage
from google.protobuf import json_format
from natsort import natsorted

from daggit.core.io.io import File_Txt
from daggit.core.io.files import findFiles
from daggit.core.base.factory import BaseOperator
from daggit.core.io.io import ReadDaggitTask_Folderpath
from daggit.contrib.sunbird.oplib.contentreuseUtils import upload_blob
from daggit.contrib.sunbird.oplib.contentreuseUtils import do_GoogleOCR
from daggit.contrib.sunbird.oplib.contentreuseUtils import download_outputjson_reponses
from daggit.contrib.sunbird.oplib.contentreuseUtils import getblob
from daggit.contrib.sunbird.oplib.contentreuseUtils import create_manifest
from daggit.contrib.sunbird.oplib.contentreuseUtils import create_toc


class OcrTextExtraction(BaseOperator):
    """

    """
    @property
    def inputs(self):
        """
        Function that the OcrTextExtraction operator defines while returning graph inputs

        :returns: Inputs to the node of the Auto tagging graph
            DS_DATA_HOME: a localpath where the folders get created
            pathTocredentials: path to config file with credentials 
            pathToPDF: path to PDF file

        """
        return {
                "DS_DATA_HOME": ReadDaggitTask_Folderpath(self.node.inputs[0]), #need to make change in the postional argument execution
                "pathTocredentials": ReadDaggitTask_Folderpath(self.node.inputs[1]),
                "pathToPDF":  ReadDaggitTask_Folderpath(self.node.inputs[2])
                }

    @property
    def outputs(self):
        """
        Function that the OcrTextExtraction operator defines while returning graph outputs

        :returns: Returns the path to the folder in which text extraction results get generated

        """
        return {"path_to_result_folder": File_Txt(
                self.node.outputs[0])}

    def run(self, gcp_bucket_name, ocr_method, content_id):

        DS_DATA_HOME = self.inputs["DS_DATA_HOME"].read_loc()
        pathTocredentials = self.inputs["pathTocredentials"].read_loc()
        path_to_PDF_file = self.inputs["pathToPDF"].read_loc()
        status = False
        print("***path to credentials:***", pathTocredentials)
        if os.path.exists(pathTocredentials):
            try:
                config = configparser.ConfigParser(allow_no_value=True)
                config.read(pathTocredentials)
                api_key = config["postman credentials"]["api_key"]
                postman_token = config["postman credentials"]["postman_token"]
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

        # content dump:
        if os.path.exists(path_to_PDF_file):
            try:
                print("Content ID: ", content_id)
                print("-------Performing OCR TextExtraction-------")
                print("path_to_pdf: ", path_to_PDF_file)
                path_to_saved_folder = ""
                start = time.time()
                path_to_saved_folder = getblob(ocr_method, gcp_bucket_name, path_to_PDF_file, content_id, DS_DATA_HOME)
                stop = time.time()
                time_consumed = stop-start
                time_consumed_minutes = time_consumed/60.0
                print("Time consumed in minutes: ", time_consumed_minutes)
                print()
            except:
                print("Error in OCR process!!")
            # Create manifest.json and content TOC
            path_to_manifest = create_manifest(content_id, path_to_saved_folder)
            path_to_toc = create_toc(content_id, path_to_saved_folder, api_key, postman_token)
        else:
            print("The path to pdf file doesnot exist!!")
        self.outputs["path_to_result_folder"].write(path_to_saved_folder)