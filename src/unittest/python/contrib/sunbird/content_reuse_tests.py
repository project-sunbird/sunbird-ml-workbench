import sys
import unittest

import pandas as pd
import os
from daggit.core.oplib.misc import df_feature_check

testdir = os.path.dirname(os.path.realpath(__file__))
srcdir = '../../../../../src/unittest/python/contrib'
os.chdir(os.path.join(testdir, '../../../'))
abs_path = os.path.abspath(os.path.join(testdir, srcdir))
test_case_data_location = abs_path + "/sunbird/test_cases_data/"
sys.path.insert(0, abs_path)


class UnitTests(unittest.TestCase):

    @staticmethod
    def test_content_resuse_scoring_data():
        cols = ['STB_Id', 'STB_Grade', 'STB_Section', 'STB_Text', 'Ref_id', 'Ref_Grade', 'Ref_Section', 'Ref_Text']
        case1 = pd.read_csv(
            test_case_data_location + "df_feature_check/" + "content_reuse_preparation_feature_check.csv")
        assert df_feature_check(case1, cols)

    @staticmethod
    def test_content_reuse_evaluation_data():
        cols = ['state_topic_id', 'reference_topic_id', 'pred_label_percentage', 'TP_count', 'FP_count', 'TN_count',
                'FN_count', 'actual_label']
        case1 = pd.read_csv(
            test_case_data_location + "df_feature_check/" + "content_reuse_evaluation_feature_check.csv")
        assert df_feature_check(case1, cols)
