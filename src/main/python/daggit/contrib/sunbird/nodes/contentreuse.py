import sys
import os
import daggit
import requests
import io
import time
import re
import logging
import configparser
import Levenshtein
import pandas as pd
import json
from scipy.spatial import distance_matrix
import ruptures as rp

from google.cloud import vision
from google.cloud import storage
from google.protobuf import json_format
from natsort import natsorted

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
from daggit.contrib.sunbird.oplib.contentreuseUtils import getDTB
from daggit.contrib.sunbird.oplib.contentreuseUtils import getSimilarTopic
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

	def run(self):

		"""
		Creates the DTB by aligning ToC with text extractd from text extracted from any textbook

		:returns: Returns the path to the DTB file

		"""
		path_to_result_folder = self.inputs["path_to_result_folder"].read()
		path_to_toc = self.inputs["pathToToc"].read()
		path_to_text = os.path.join(path_to_result_folder, "extract", "GOCR", "text", "fulltext_annotation.txt")
		dtb = create_dtb(path_to_toc, path_to_text)
		path_to_dtb_json_file = os.path.join(path_to_result_folder, "DTB.json")
		with open(path_to_dtb_json_file, "w") as outfile:
			json.dump(dtb, outfile, indent=4)
		self.outputs["path_to_dtb_json_file"].write(path_to_dtb_json_file)


class DTBMapping(BaseOperator):
	
	@property
	def inputs(self):
		"""
		Function that the DTBMapping operator defines while returning graph inputs

		:returns: Inputs to the node of the Auto tagging graph
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
