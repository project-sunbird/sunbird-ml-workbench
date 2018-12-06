from contrib.sunbird.TestingUtils import text_reading
from contrib.sunbird.TestingUtils import sentence_similarity
from contrib.sunbird.TestingUtils import keyword_extraction
import unittest
import sys
import os
import yaml
import pandas as pd
from daggit.contrib.sunbird.operators.contentTaggingUtils import *

testdir = os.path.dirname(os.path.realpath(__file__))
srcdir = '../../../../../src/unittest/python/contrib'
os.chdir(os.path.join(testdir, '../../../'))
abs_path = os.path.abspath(os.path.join(testdir, srcdir))
test_case_data_location = abs_path + "/sunbird/test_cases_data/"
sys.path.insert(0, abs_path)


class UnitTests(unittest.TestCase):

    @staticmethod
    def test_df_feature_check():
        case1 = pd.read_csv(
            test_case_data_location +
            "df_feature_check/" +
            "Content_Meta_feature_checking_df_1.csv")
        case2 = pd.read_csv(
            test_case_data_location +
            "df_feature_check/" +
            "Content_Meta_feature_checking_df_2.csv")
        case3 = pd.read_csv(
            test_case_data_location +
            "df_feature_check/" +
            "Content_Meta_feature_checking_df_3.csv")
        mandatatory_field_location = test_case_data_location + \
            "df_feature_check/" + "ContentTagging_mandatory_fields.yaml"
        with open(mandatatory_field_location, 'r') as stream:
            data = yaml.load(stream)
        mandatatory_field_ls = list(data['mandatory_fields'])
        assert df_feature_check(case1, mandatatory_field_ls)
        assert df_feature_check(case2, mandatatory_field_ls) == False
        assert df_feature_check(case3, mandatatory_field_ls) == False

    @staticmethod
    def test_speech_to_text():
        actual_text = speech_to_text(
            'googleAT',
            test_case_data_location +
            "SpeechText/id_1/assets")["text"]
        expected_text = text_reading(
            test_case_data_location +
            "SpeechText/" +
            'speech_to_text_exp_output.txt')
        assert sentence_similarity(actual_text, expected_text, .70) == 1

    @staticmethod
    def test_pdf_to_text():
        case1_actual_text = pdf_to_text(
            "PyPDF2",
            test_case_data_location +
            "PdfText/id_1",
            'none')["text"]
        case1_expected_text = text_reading(
            test_case_data_location + "PdfText/id_1/" + 'ExpText.txt')
        case2_actual_text = pdf_to_text(
            "PyPDF2",
            test_case_data_location +
            "PdfText/id_2",
            'none')["text"]
        case2_expected_text = text_reading(
            test_case_data_location + "PdfText/id_2/" + 'ExpText.txt')
        assert sentence_similarity(
            case1_actual_text, case1_expected_text, .70) == 1
        assert sentence_similarity(
            case2_actual_text, case2_expected_text, .70) == 1

    @staticmethod
    def test_keyword_extraction():
        eng_text_actual_keywords = pd.read_csv(
            test_case_data_location +
            "keyword_extraction/" +
            "eng_text_actual_keywords.csv")['KEYWORDS']
        assert keyword_extraction(
            test_case_data_location +
            "keyword_extraction/" +
            "empty.txt",
            list(eng_text_actual_keywords)) == "Text is not available"
        assert keyword_extraction(
            test_case_data_location +
            "keyword_extraction/" +
            "english.txt",
            list(eng_text_actual_keywords)) == 1

    @staticmethod
    def test_jaccard_evaluation():
        assert jaccard_with_phrase(['simple', 'algebraic', 'problem', 'mathematics'], [
                                   'algebraic', 'maths'])['jaccard'] == 0.2
        assert jaccard_with_phrase(['simple', 'algebraic', 'problem', 'mathematics'], [
                                   'tree', 'apple'])['jaccard'] == 0
