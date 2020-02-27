import sys
import os
import daggit
import requests
import io
import time
import re
import glob
import logging
import pyvips
import cv2
import img2pdf
import configparser
import Levenshtein
import pandas as pd
import json
from scipy.spatial import distance_matrix
import ruptures as rp
import pandas as pd
import numpy as np 

from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path

from google.cloud import vision
from google.cloud import storage
from google.protobuf import json_format
from natsort import natsorted, ns

from daggit.core.io.io import File_IO 
from daggit.core.io.files import findFiles
from daggit.core.base.factory import BaseOperator
from daggit.core.io.io import ReadDaggitTask_Folderpath
from daggit.contrib.sunbird.oplib.contentreuseUtils import upload_blob
from daggit.contrib.sunbird.oplib.contentreuseUtils import do_GoogleOCR
from daggit.contrib.sunbird.oplib.contentreuseUtils import download_outputjson_reponses
from daggit.contrib.sunbird.oplib.contentreuseUtils import getblob
from daggit.contrib.sunbird.oplib.contentreuseUtils import create_manifest
from daggit.contrib.sunbird.oplib.contentreuseUtils import create_toc
from daggit.contrib.sunbird.oplib.dtb import create_dtb
from daggit.contrib.sunbird.oplib.contentreuseUtils import getDTB, calc_stat
from daggit.contrib.sunbird.oplib.contentreuseUtils import agg_actual_predict_df
from daggit.contrib.sunbird.oplib.contentreuseUtils import getSimilarTopic
from daggit.contrib.sunbird.oplib.contentreuseEvaluationUtils import text_image
from daggit.core.oplib import distanceUtils as dist
from daggit.core.oplib import nlp as preprocess


class OcrTextExtraction(BaseOperator):
    
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
                "DS_DATA_HOME": File_IO(self.node.inputs[0]),
                "pathTocredentials": File_IO(self.node.inputs[1]),
                "pathToPDF":  File_IO(self.node.inputs[2])
                }

    @property
    def outputs(self):
        """
        Function that the OcrTextExtraction operator defines while returning graph outputs

        :returns: Returns the path to the folder in which text extraction results get generated

        """
        return {"path_to_result_folder": File_IO(
                self.node.outputs[0])}

    def run(self, gcp_bucket_name, ocr_method, content_id):

        DS_DATA_HOME = self.inputs["DS_DATA_HOME"].read()
        pathTocredentials = self.inputs["pathTocredentials"].read()
        path_to_PDF_file = self.inputs["pathToPDF"].read()
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


class TextExtractionEvaluation(BaseOperator):
    
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
                "path_to_result_folder": File_IO(self.node.inputs[0]),
                "pathToLanguageMapping": File_IO(self.node.inputs[1]),
                "pathToToc": File_IO(self.node.inputs[2])
                }

    @property
    def outputs(self):
        """
        Function that the OcrTextExtraction operator defines while returning graph outputs

        :returns: Returns the path to the folder in which text extraction results get generated
        """
        return {"path_to_validation_pdf": File_IO(
                self.node.outputs[0])}

    def run(self):
        target_folder = self.inputs["path_to_result_folder"].read()
        language_mapping_loc = self.inputs["pathToLanguageMapping"].read()
        path_to_toc = self.inputs["pathToToc"].read()
        evaluation_folder = os.path.join(target_folder, "evaluation")
        if not os.path.exists(evaluation_folder):
            os.makedirs(evaluation_folder)
        output_loc = os.path.join(evaluation_folder, "text_extraction_validation")
        if not os.path.exists(output_loc):
            os.mkdir(output_loc)
        pdf_files = [i for i in os.listdir(os.path.join(target_folder, "source")) if i!='.DS_Store']
        pdf_ocr_loc_ls = [] 
        for i in range(len(pdf_files)):
            pdf_loc = os.path.join(target_folder, "source", pdf_files[i])  
            print(pdf_loc)
            google_ocr_loc = os.path.join(target_folder, "raw_data", os.path.split(target_folder)[1]) 
            pdf_ocr_loc_ls.append([pdf_loc, google_ocr_loc])
        toc_df = pd.read_csv(path_to_toc)
        medium = [i for i in toc_df["Medium"].unique() if str(i)!="nan"][0]
        pdf_ocr_loc_ls
        for i in range(len(pdf_ocr_loc_ls)):
            text_image(pdf_ocr_loc_ls[i][0],pdf_ocr_loc_ls[i][1],medium, language_mapping_loc, 1, output_loc)

        image_ls = [i for i in os.listdir(output_loc) if i.endswith(".png")] 
        ls = [i for i in natsorted(image_ls, reverse=False)] 
        image_ls_open = [Image.open(os.path.join(output_loc,str(i))) for i in ls ]
        pdf_filename = os.path.join(output_loc, "validation.pdf")

        image_ls_open[0].save(pdf_filename, "PDF" ,resolution=100.0, save_all=True, append_images=image_ls_open[1:]) 
        for files in image_ls:
             os.remove(os.path.join(output_loc, str(files)))
        self.outputs["path_to_validation_pdf"].write(pdf_filename)


class CreateDTB(BaseOperator):
    
    @property
    def inputs(self):
        """
        Inputs needed to create DTB

        :returns: toc and text files

        """
        return {
                "path_to_result_folder": File_IO(self.node.inputs[0]),
                "pathToToc": File_IO(self.node.inputs[1]),
                }

    @property
    def outputs(self):
        """
        Outputs created while creating DTB

        :returns: Returns the path to the DTB file

        """
        return {"path_to_dtb_json_file": File_IO(
                self.node.outputs[0])}

    def run(self, col_name):

        """
        Creates the DTB by aligning ToC with text extractd from text extracted from any textbook

        :returns: Returns the path to the DTB file

        """
        path_to_result_folder = self.inputs["path_to_result_folder"].read()
        path_to_toc = self.inputs["pathToToc"].read()
        path_to_text = os.path.join(path_to_result_folder, "extract", "GOCR", "text", "fulltext_annotation.txt")
        dtb = create_dtb(path_to_toc, path_to_text, col_name)
        path_to_dtb_json_file = os.path.join(path_to_result_folder, "DTB.json")
        with open(path_to_dtb_json_file, "w") as outfile:
            json.dump(dtb, outfile, indent=4)
        self.outputs["path_to_dtb_json_file"].write(path_to_dtb_json_file)


class DTBCreationEvaluation(BaseOperator):
    
    @property
    def inputs(self):
        """
        Function that the DTBCreationEvaluation operator defines while returning graph inputs

        :returns: Inputs to the node of the Content Reuse graph
            path_to_result_folder: path to result folder
            path_to_dtb_json_file: path to predicted DTB file
            pathToToc: path to TOC
            pathToActualDTB: path to actual DTB

        """
        return {
                "path_to_dtb_json_file": File_IO(self.node.inputs[0]),
                "pathToToc": File_IO(self.node.inputs[1]),
                "pathToActualDTB": File_IO(self.node.inputs[2])
                }

    @property
    def outputs(self):
        """
        Function that the DTBMapping operator defines while returning graph outputs

        :returns: Returns the path to the mapping json file

        """
        return {"path_to_dtb_evaluation_result": File_IO(
                self.node.outputs[0])}
    
    def run(self, level):
        dtb_pred_loc = self.inputs['path_to_dtb_json_file'].read()
        assert os.path.exists(dtb_pred_loc) == True
        path_to_result_folder = os.path.split(dtb_pred_loc)[0]
        evaluation_folder = os.path.join(path_to_result_folder, "evaluation")
        if not os.path.exists(evaluation_folder):
            os.makedirs(evaluation_folder)
        text_loc = os.path.join(path_to_result_folder, "extract", "GOCR", "text", "fulltext_annotation.txt")
        dtb_actual_loc = self.inputs['pathToActualDTB'].read()
        toc_df_loc = self.inputs['pathToToc'].read()
        output_loc = os.path.join(evaluation_folder, "DTB_creation_evaluation.csv")

        toc_df = pd.read_csv(toc_df_loc) 
        dtb_actual = pd.read_csv(dtb_actual_loc)
        if 'Toc feature' in dtb_actual.columns:
            with open(text_loc, 'r') as txt_file:
                text = txt_file.read()
            with open(dtb_pred_loc, 'r') as f:
                dtb_predicted = json.load(f)
            text_read_ = preprocess.strip_word_number([text], " ")[0]    
            text_read_ = re.sub(' +', ' ', text_read_)
            toc_df[['Chapter Name','Topic Name']] = toc_df[['Chapter Name','Topic Name']].apply(lambda x: preprocess.strip_word_number(x, " ")) 
            toc_df = pd.DataFrame(toc_df.groupby('Chapter Name')['Topic Name'].unique())
            pred_df = pd.DataFrame( )
            pred_df['title'] = [dtb_predicted['alignment'][i]['source']['fulltext_annotation'] for i in range(len(dtb_predicted['alignment']))] 
            pred_df['pred_text']=[ dtb_predicted['alignment'][i]['target']['fulltext_annotation'] for i in range(len(dtb_predicted['alignment'])) ] 
            pred_df[['title','pred_text']] = pred_df[['title','pred_text']].apply(lambda x: preprocess.strip_word_number(x, " "))
            dtb_actual[['CONTENTS','Toc feature']] = dtb_actual[['CONTENTS','Toc feature']].apply(lambda x: preprocess.strip_word_number(x, " "))
            actual_predict_df_ls = agg_actual_predict_df(toc_df,dtb_actual,pred_df, level)
            concat_df = pd.concat(actual_predict_df_ls)
            concat_df = concat_df.reset_index()
            concat_df['Actual_text_split'] = [set(i.split()) for i in concat_df['ActualText']] 
            concat_df['Pred_text_split'] = [set(i.split()) for i in concat_df['PredictedText']]  
            concat_df['Common_words'] = [set(concat_df['Actual_text_split'][i])&set(concat_df['Pred_text_split'][i]) for i in range(len(concat_df))]  
            concat_df['Len_actual_text_split'] = [ float(len(i)) for i in list(concat_df['Actual_text_split'])] 
            concat_df['Len_pred_text_split'] = [ float(len(i)) for i in list(concat_df['Pred_text_split'])] 
            concat_df['Len_common_words'] = [ float(len(i)) for i in list(concat_df['Common_words'])] 
            concat_df['Intersection/actu'] = calc_stat(list(concat_df['Common_words']), list(concat_df['Actual_text_split']), "division")
            concat_df['Intersection/pred'] = calc_stat(list(concat_df['Common_words']), list(concat_df['Pred_text_split']), "division")
            concat_df['WordlistEMD'] = calc_stat(list(concat_df['Actual_text_split']), list(concat_df['Pred_text_split']), "MED") 
            concat_df.to_csv(output_loc, index = False)
            self.outputs["path_to_dtb_evaluation_result"].write(output_loc)
        else:
            print("The column is not present in the Dataframe!!")


class DTBMapping(BaseOperator):
    
    @property
    def inputs(self):
        """
        Function that the DTBMapping operator defines while returning graph inputs

        :returns: Inputs to the node of the Content Reuse graph
            path_to_result_folder: path to result folder
            path_to_reference_DTB: path to reference DTB

        """
        return {
                "path_to_result_folder": File_IO(self.node.inputs[0]),
                "path_to_dtb_json_file": File_IO(self.node.inputs[1]),
                "path_to_reference_DTB": File_IO(self.node.inputs[2])
                }

    @property
    def outputs(self):
        """
        Function that the DTBMapping operator defines while returning graph outputs

        :returns: Returns the path to the mapping json file

        """
        return {"path_to_mapping_json": File_IO(
                self.node.outputs[0])}

    def run(self, no_of_recommendations, distance_method):

        path_to_result_folder = self.inputs["path_to_result_folder"].read()
        path_to_dtb_json_file = self.inputs["path_to_dtb_json_file"].read()
        path_to_ref_dtb = self.inputs["path_to_reference_DTB"].read()

        path_to_mapping_json = os.path.join(path_to_result_folder, distance_method+"_mapping.json")
        state_DTB = getDTB(path_to_dtb_json_file)
        reference_DTB = pd.DataFrame()
        for TB in os.listdir(path_to_ref_dtb):
            try:
                DTB_df = getDTB(os.path.join(path_to_ref_dtb, TB, "DTB.json"))
                DTB_df["textbook"] = TB
                reference_DTB = reference_DTB.append(DTB_df)
            except:
                pass
        reference_DTB = reference_DTB.reset_index()

        if distance_method == "BERT":
            distance = dist.getDistance(list(state_DTB['text']), list(reference_DTB['text']), 'BERT')
        elif distance_method == "WMD":
            distance =dist.getDistance(list(state_DTB['text']), list(reference_DTB['text']), 'WMD')
        else:
            print("Invalid distance measure!!")
        try:
            distance_df = pd.DataFrame(distance, index= list(state_DTB['identifier']), columns= list(reference_DTB['identifier']))
            topN_similar = getSimilarTopic(distance_df, no_of_recommendations)

            json.dump(topN_similar, open(path_to_mapping_json, "w"), indent=4)
            self.outputs["path_to_mapping_json"].write(path_to_mapping_json)
        except:
            print("Distance not computed")


