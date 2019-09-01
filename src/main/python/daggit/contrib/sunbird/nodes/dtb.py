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


from daggit.core.io.io import File_Txt
from daggit.core.io.io import File_JSON
from daggit.core.io.io import ReadDaggitTask_Folderpath
from daggit.core.base.factory import BaseOperator

from daggit.contrib.sunbird.oplib.dtb import create_dtb



class CreateDTB(BaseOperator):
	
	@property
	def inputs(self):
		"""
		Inputs needed to create DTB

		:returns: toc and text files

		"""
		return {
				"toc_csv_file": ReadDaggitTask_Folderpath(self.node.inputs[0]),
				"extract_text_file": ReadDaggitTask_Folderpath(self.node.inputs[1]),
				}

	@property
	def outputs(self):
		"""
		Outputs created while creating DTB

		:returns: Returns the path to the DTB file

		"""
		return {"dtb_json_file": File_JSON(
				self.node.outputs[0])}

	def run(self):

		"""
		Creates the DTB by aligning ToC with text extractd from text extracted from any textbook

		:returns: Returns the path to the DTB file

		"""

		f_toc = self.inputs["toc_csv_file"].read_loc()
		f_text = self.inputs["extract_text_file"].read_loc()
		dtb = create_dtb(f_toc,f_text)
		self.outputs["dtb_json_file"].write(dtb)