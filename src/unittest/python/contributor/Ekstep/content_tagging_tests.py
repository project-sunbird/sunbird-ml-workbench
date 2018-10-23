import unittest
import sys, os
import pandas as pd


testdir = os.path.dirname(os.path.realpath(__file__))
srcdir = '../../../../../src/unittest/python/contributor' 
os.chdir(os.path.join(testdir, '../../../'))
abs_path = os.path.abspath(os.path.join(testdir, srcdir)) 
test_case_data_location = abs_path + "/Ekstep/test_cases_data/" 
sys.path.insert(0, abs_path)

from contributor.Ekstep.TestingUtils import content_meta_features_checking
from contributor.Ekstep.TestingUtils import text_Extraction
from contributor.Ekstep.TestingUtils import keyword_extraction
from contributor.Ekstep.TestingUtils import jaccard_evaluation



class UnitTests(unittest.TestCase):
	
	@staticmethod
	def test_content_meta_features_checking():
		content_meta_location = test_case_data_location + "Content_Meta_feature_checking_df_1.csv"
		mandatatory_field_location = test_case_data_location + "ContentTagging_mandatory_fields.yaml" 
		assert content_meta_features_checking(content_meta_location, mandatatory_field_location) == 1
	
	@staticmethod
	def test_keyword_extraction():
		assert keyword_extraction(test_case_data_location + "empty.txt", test_case_data_location, ["animal"])== "Text is not available"
	
	@staticmethod
	def test_text_Extraction():
		assert text_Extraction('https://www.youtube.com/embed/ypxU42bbqWw?autoplay=1&enablejsapi=1', 'youtube', 'id_1', test_case_data_location , "plants Baje a little seed for me to sow a little app to make it grow a little help a little thought on the Beach and that is that a little son a little shower and a little better") == 1
	
	@staticmethod
	def test_jaccard_evaluation():
	    assert jaccard_evaluation(['simple','algebraic','problem','mathematics'],['algebraic','maths'],'jaccard', 0.6 ) == 1 


