import unittest
import sys, os
import yaml
import pandas as pd
import daggit
from daggit.contrib.ekstep.operators.contentTaggingUtils import *

testdir = os.path.dirname(os.path.realpath(__file__))
srcdir = '../../../../../src/unittest/python/contrib' 
os.chdir(os.path.join(testdir, '../../../'))
abs_path = os.path.abspath(os.path.join(testdir, srcdir)) 
test_case_data_location = abs_path + "/ekstep/test_cases_data/" 
sys.path.insert(0, abs_path)

from contrib.ekstep.TestingUtils import keyword_extraction
from contrib.ekstep.TestingUtils import jaccard_evaluation
from contrib.ekstep.TestingUtils import sentence_similarity


class UnitTests(unittest.TestCase):
	
	@staticmethod
	def test_df_feature_check():
		test_df1 = pd.read_csv(test_case_data_location + "Content_Meta_feature_checking_df_1.csv")
		test_df2 = pd.read_csv(test_case_data_location + "Content_Meta_feature_checking_df_2.csv")
		test_df3 = pd.read_csv(test_case_data_location + "Content_Meta_feature_checking_df_3.csv") 
		mandatatory_field_location = test_case_data_location + "ContentTagging_mandatory_fields.yaml"
		with open(mandatatory_field_location, 'r') as stream:
			data = yaml.load(stream)
		mandatatory_field_ls = list(data['mandatory_fields'])
		assert df_feature_check(test_df1,mandatatory_field_ls) == True
		assert df_feature_check(test_df2,mandatatory_field_ls) == False
		assert df_feature_check(test_df3,mandatatory_field_ls) == False

	@staticmethod
	def test_speech_to_text():
		actual_text = speech_to_text('googleAT',test_case_data_location)
		file = open(test_case_data_location + 'speech_to_text_exp_output.txt', "r")
		expected_text = file.readline()
		assert sentence_similarity(actual_text,expected_text,.90)== 1

	@staticmethod
	def test_pdf_to_text():
		actual_text = pdf_to_text("PyPDF2", "/Users/anjana/ML-Workbench/src/unittest/python/contrib/ekstep/test_cases_data/assets", 'none') 
		file = open(test_case_data_location + 'pdf_to_text_exp_output.txt', "r")
		expected_text = file.readline()
		assert sentence_similarity(actual_text,expected_text,.50)== 1
	
	
	
	@staticmethod
	def test_keyword_extraction():
		eng_text_actual_keywords = pd.read_csv(test_case_data_location + "eng_text_actual_keywords.csv")['KEYWORDS']
		assert keyword_extraction(test_case_data_location + "empty.txt", test_case_data_location, list(eng_text_actual_keywords))== "Text is not available"
		assert keyword_extraction(test_case_data_location + "english.txt", test_case_data_location, list(eng_text_actual_keywords)) == 1 

	@staticmethod
	def test_jaccard_evaluation():
	    assert jaccard_evaluation(['simple','algebraic','problem','mathematics'],['algebraic','maths'],'jaccard', 0.6 ) == 1 


