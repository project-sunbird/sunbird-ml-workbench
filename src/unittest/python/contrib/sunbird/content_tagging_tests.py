import sys
import unittest
import daggit
from daggit.contrib.sunbird.oplib.taggingUtils import *
from daggit.core.oplib.misc import df_feature_check
from daggit.core.oplib.nlp import jaccard_with_phrase
from nltk.corpus import stopwords
# from daggit.contrib.sunbird.oplib.contentreuseUtils import scoring_module, filter_by_grade_range
# from daggit.contrib.sunbird.oplib.contentreuseUtils import aggregation_topic_level

stopwords = stopwords.words('english')

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

    # @staticmethod
    # def test_speech_to_text():
    #     try:
    #         GOOGLE_APPLICATION_CREDENTIALS = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
    #         print("******GOOGLE_APPLICATION_CREDENTIALS :", GOOGLE_APPLICATION_CREDENTIALS)
    #         with open(GOOGLE_APPLICATION_CREDENTIALS, "r") as f:
    #             GOOGLE_APPLICATION_CREDENTIALS = f.read()
    #         actual_text = speech_to_text(
    #             'googleAT',
    #             test_case_data_location +
    #             "SpeechText/id_1/assets", GOOGLE_APPLICATION_CREDENTIALS)["text"]
    #         expected_text = text_reading(
    #             test_case_data_location +
    #             "SpeechText/" +
    #             'speech_to_text_exp_output.txt')
    #         assert sentence_similarity(actual_text, expected_text, .70) == 1
    #     except BaseException:
    #         print("Unable to retrieve environment variable. Check if GOOGLE_APPLICATION_CREDENTIALS  is set")

    # @staticmethod
    # def test_pdf_to_text():
    #     case1_actual_text = pdf_to_text(
    #         "pdfminer",
    #         test_case_data_location +
    #         "PdfText/id_1",
    #         'none')["text"]
    #     case1_expected_text = text_reading(
    #         test_case_data_location + "PdfText/id_1/" + 'ExpText.txt')
    #     case2_actual_text = pdf_to_text(
    #         "pdfminer",
    #         test_case_data_location +
    #         "PdfText/id_2",
    #         'none')["text"]
    #     case2_expected_text = text_reading(
    #         test_case_data_location + "PdfText/id_2/" + 'ExpText.txt')
    #     assert sentence_similarity(
    #         case1_actual_text, case1_expected_text, .70) == 1
    #     assert sentence_similarity(
    #         case2_actual_text, case2_expected_text, .70) == 1

    # @staticmethod
    # def test_keyword_extraction():
    #     eng_text_actual_keywords = pd.read_csv(
    #         test_case_data_location +
    #         "keyword_extraction/" +
    #         "eng_text_actual_keywords.csv")['KEYWORDS']
    #     assert keyword_extraction(
    #         test_case_data_location +
    #         "keyword_extraction/" +
    #         "empty.txt",
    #         list(eng_text_actual_keywords)) == "Text is not available"
    #     assert keyword_extraction(
    #         test_case_data_location +
    #         "keyword_extraction/" +
    #         "english.txt",
    #         list(eng_text_actual_keywords)) == 1

    @staticmethod
    def test_jaccard_evaluation():
        assert jaccard_with_phrase(['simple', 'algebraic', 'problem', 'mathematics'], [
            'algebraic', 'maths'])['jaccard'] == 0.2
        assert jaccard_with_phrase(['simple', 'algebraic', 'problem', 'mathematics'], [
            'tree', 'apple'])['jaccard'] == 0


    @staticmethod
    def test_scoring_data_preparation():
        cols = ['STB_Id', 'STB_Grade', 'STB_Section', 'STB_Text', 'Ref_id', 'Ref_Grade', 'Ref_Section', 'Ref_Text']
        case1 = pd.read_csv(test_case_data_location + "df_feature_check/" + "content_reuse_feature_check.csv")
        assert df_feature_check(case1, cols)


    # @staticmethod
    # def test_bert_scoring():
    #     test_data = pd.read_csv(test_case_data_location + "bert_scoring/" + "input.csv")
    #     test_tokenizer_path = pd.read_csv(
    #         test_case_data_location +
    #         "bert_scoring/" +
    #         "tokenizer.pkl")
    #     test_model_path = pd.read_csv(
    #         test_case_data_location +
    #         "bert_scoring/" +
    #         "lstm_50_50_0_17_0_25.h5")
    #     test_config = pd.read_csv(
    #         test_case_data_location +
    #         "bert_scoring/" +
    #         "siamese_configuration.json")
    #     test_threshold = 0.482528924942016
    #     with open(test_tokenizer_path, 'rb') as tokenizer_file:
    #         test_tokenizer = pickle.load(tokenizer_file)
    #     output_pred_df = scoring_module(test_tokenizer, test_model_path, test_config, test_data, test_threshold)
    #     mandatory_field_location = test_case_data_location + "bert_scoring/" + "bert_scoring_mandatory_fields.yaml"
    #     with open(mandatory_field_location, 'r') as stream:
    #         data = yaml.load(stream)
    #     mandatory_field_ls = list(data['mandatory_fields'])
    #     assert df_feature_check(output_pred_df, mandatory_field_ls)
    #
    # @staticmethod
    # def test_aggregation_topic_level():
    #     test_output_bert = pd.read_csv(test_case_data_location + "bert_scoring/" + "input.csv")
    #     aggregation_criteria = "average"
    #     mandatory_field_location = test_case_data_location + "bert_scoring/" + "topic_aggregation_mandatory_fields.yaml"
    #     with open(mandatory_field_location, 'r') as stream:
    #         data = yaml.load(stream)
    #     mandatory_field_ls = list(data['mandatory_fields'])
    #     output_aggregated_topic_level = aggregation_topic_level(test_output_bert,
    #                                                             aggregation_criteria, data["mandatory_column_name"])
    #     assert df_feature_check(output_aggregated_topic_level, mandatory_field_ls)
    #
    #
    #
