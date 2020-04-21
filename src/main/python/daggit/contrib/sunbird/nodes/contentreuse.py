import ast
import configparser
import json
import logging
import os
# import pyvips
import pickle
import re
import time

import pandas as pd
from PIL import Image
from daggit.contrib.sunbird.oplib.contentreuseUtils import agg_actual_predict_df
from daggit.contrib.sunbird.oplib.contentreuseUtils import aggregation_topic_level
from daggit.contrib.sunbird.oplib.contentreuseUtils import append_cosine_similarity_score
from daggit.contrib.sunbird.oplib.contentreuseUtils import connect_to_graph, create_node_relationships
from daggit.contrib.sunbird.oplib.contentreuseUtils import create_manifest
from daggit.contrib.sunbird.oplib.contentreuseUtils import create_toc
from daggit.contrib.sunbird.oplib.contentreuseUtils import generate_cosine_similarity_score
from daggit.contrib.sunbird.oplib.contentreuseUtils import getDTB, calc_stat
from daggit.contrib.sunbird.oplib.contentreuseUtils import getSimilarTopic
from daggit.contrib.sunbird.oplib.contentreuseUtils import getblob
from daggit.contrib.sunbird.oplib.contentreuseUtils import k_topic_recommendation
from daggit.contrib.sunbird.oplib.contentreuseUtils import modify_df
from daggit.contrib.sunbird.oplib.contentreuseUtils import scoring_module, filter_by_grade_range
from daggit.contrib.sunbird.oplib.dtb import create_dtb
from daggit.core.base.factory import BaseOperator
from daggit.core.io.io import File_IO, File_Txt
from daggit.core.io.io import ReadDaggitTask_Folderpath
# from daggit.contrib.sunbird.oplib.contentreuseEvaluationUtils import text_image
from daggit.core.oplib import distanceUtils as dist
from daggit.core.oplib import nlp as preprocess
from natsort import natsorted


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
            "pathToPDF": File_IO(self.node.inputs[2])
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
                time_consumed = stop - start
                time_consumed_minutes = time_consumed / 60.0
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
        pdf_files = [i for i in os.listdir(os.path.join(target_folder, "source")) if i != '.DS_Store']
        pdf_ocr_loc_ls = []
        for i in range(len(pdf_files)):
            pdf_loc = os.path.join(target_folder, "source", pdf_files[i])
            print(pdf_loc)
            google_ocr_loc = os.path.join(target_folder, "raw_data", os.path.split(target_folder)[1])
            pdf_ocr_loc_ls.append([pdf_loc, google_ocr_loc])
        toc_df = pd.read_csv(path_to_toc)
        medium = [i for i in toc_df["Medium"].unique() if str(i) != "nan"][0]
        for i in range(len(pdf_ocr_loc_ls)):
            text_image(pdf_ocr_loc_ls[i][0], pdf_ocr_loc_ls[i][1], medium, language_mapping_loc, 1, output_loc)
        image_ls = [i for i in os.listdir(output_loc) if i.endswith(".png")]
        ls = [i for i in natsorted(image_ls, reverse=False)]
        image_ls_open = [Image.open(os.path.join(output_loc, str(i))) for i in ls]
        pdf_filename = os.path.join(output_loc, "validation.pdf")

        image_ls_open[0].save(pdf_filename, "PDF", resolution=100.0, save_all=True, append_images=image_ls_open[1:])
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
            toc_df[['Chapter Name', 'Topic Name']] = toc_df[['Chapter Name', 'Topic Name']].apply(
                lambda x: preprocess.strip_word_number(x, " "))
            toc_df = pd.DataFrame(toc_df.groupby('Chapter Name')['Topic Name'].unique())
            pred_df = pd.DataFrame()
            pred_df['title'] = [dtb_predicted['alignment'][i]['source']['fulltext_annotation'] for i in
                                range(len(dtb_predicted['alignment']))]
            pred_df['pred_text'] = [dtb_predicted['alignment'][i]['target']['fulltext_annotation'] for i in
                                    range(len(dtb_predicted['alignment']))]
            pred_df[['title', 'pred_text']] = pred_df[['title', 'pred_text']].apply(
                lambda x: preprocess.strip_word_number(x, " "))
            dtb_actual[['CONTENTS', 'Toc feature']] = dtb_actual[['CONTENTS', 'Toc feature']].apply(
                lambda x: preprocess.strip_word_number(x, " "))
            actual_predict_df_ls = agg_actual_predict_df(toc_df, dtb_actual, pred_df, level)
            concat_df = pd.concat(actual_predict_df_ls)
            concat_df = concat_df.reset_index()
            concat_df['Actual_text_split'] = [set(i.split()) for i in concat_df['ActualText']]
            concat_df['Pred_text_split'] = [set(i.split()) for i in concat_df['PredictedText']]
            concat_df['Common_words'] = [set(concat_df['Actual_text_split'][i]) & set(concat_df['Pred_text_split'][i])
                                         for i in range(len(concat_df))]
            concat_df['Len_actual_text_split'] = [float(len(i)) for i in list(concat_df['Actual_text_split'])]
            concat_df['Len_pred_text_split'] = [float(len(i)) for i in list(concat_df['Pred_text_split'])]
            concat_df['Len_common_words'] = [float(len(i)) for i in list(concat_df['Common_words'])]
            concat_df['Intersection/actu'] = calc_stat(list(concat_df['Common_words']),
                                                       list(concat_df['Actual_text_split']), "division")
            concat_df['Intersection/pred'] = calc_stat(list(concat_df['Common_words']),
                                                       list(concat_df['Pred_text_split']), "division")
            concat_df['WordlistEMD'] = calc_stat(list(concat_df['Actual_text_split']),
                                                 list(concat_df['Pred_text_split']), "MED")
            concat_df.to_csv(output_loc, index=False)
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

        path_to_mapping_json = os.path.join(path_to_result_folder, distance_method + "_mapping.json")
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
            distance = dist.getDistance(list(state_DTB['text']), list(reference_DTB['text']), 'WMD')
        else:
            print("Invalid distance measure!!")
        try:
            distance_df = pd.DataFrame(distance, index=list(state_DTB['identifier']),
                                       columns=list(reference_DTB['identifier']))
            topN_similar = getSimilarTopic(distance_df, no_of_recommendations)

            json.dump(topN_similar, open(path_to_mapping_json, "w"), indent=4)
            self.outputs["path_to_mapping_json"].write(path_to_mapping_json)
        except:
            print("Distance not computed")


class scoring_data_preparation(BaseOperator):
    @property
    def inputs(self):
        """
        Function that the ScoringDataPreparation operator defines while returning graph inputs
        :return: Inputs to the node of the Content Reuse graph
            path_to_base_data_file: path to base data file
        """
        return {
            "path_to_baseref_data": File_IO(self.node.inputs[0])
        }

    @property
    def outputs(self):
        """
        Function that the ScoringDataPreparation operator defines while returning graph outputs
        :return: Returns the path to the cosine similarity pickle and complete data set
        """
        return {
            "path_to_result_folder": File_IO(self.node.outputs[0]),
            "path_to_cosine_similarity_matrix": File_IO(self.node.outputs[1]),
            "path_to_complete_data_set": File_IO(self.node.outputs[2])
        }

    def run(self, sentence_length, cosine_score_threshold):
        """
        Generate data for the purpose of scoring
        :param sentence_length: filter data on minimum number of sentences per topic
        :param cosine_score_threshold: threshold to filter cosine similarity score on
        :return: None
        """
        path_to_result_folder = self.outputs["path_to_result_folder"].read()
        path_to_baseref_data = self.inputs["path_to_baseref_data"].read()
        print("*****path_to_result_folder: ", path_to_result_folder)
        if not os.path.exists(path_to_result_folder):
            raise Exception("Data folder not present.")
        # file_path = os.path.join(path_to_result_folder, "base_ref_general_data_prep.csv")
        try:
            base_df = pd.read_csv(path_to_baseref_data)[
                ['STB_Id', 'STB_Grade', 'STB_Section', 'STB_Text', 'Ref_id', 'Ref_Grade', 'Ref_Section', 'Ref_Text']]
        except FileNotFoundError:
            raise Exception("Base data file does not exist")
        except KeyError:
            raise Exception("Column names are invalid")
        stb_df, ref_df = modify_df(base_df, sentence_length)
        cos_sim_df = generate_cosine_similarity_score(stb_df, ref_df, path_to_result_folder)
        append_cosine_similarity_score(stb_df, ref_df, cos_sim_df, path_to_result_folder, cosine_score_threshold)
        self.outputs["path_to_cosine_similarity_matrix"].write(os.path.join(path_to_result_folder,
                                                                            'cosine_similarity.pkl'))
        self.outputs["path_to_complete_data_set"].write(os.path.join(path_to_result_folder, 'complete_data_set.csv'))


class bert_scoring(BaseOperator):
    @property
    def inputs(self):
        """
        Function that the BERTScoring operator defines while returning graph inputs

        :returns: Inputs to the node of the Content Reuse graph
            path_to_result_folder: path to result folder
            path_to_reference_DTB: path to reference DTB

        """
        return {
            "path_to_trained_model": ReadDaggitTask_Folderpath(self.node.inputs[0]),
            "path_to_pickled_tokenizer": File_IO(self.node.inputs[1]),
            "path_to_scoring_data": File_IO(self.node.inputs[2]),
            "path_to_siamese_config": File_IO(self.node.inputs[3])
        }

    @property
    def outputs(self):
        """
        Function that the BERTScoring operator defines while returning graph outputs

        :returns: Returns the path to the mapping json file

        """
        return {
            "path_to_predicted_output": File_Txt(
            self.node.outputs[0])}

    def run(self, filterby_typeofmatch, filterby_grade_range, threshold, embedding_method):
        path_to_best_model = self.inputs["path_to_trained_model"].read_loc()
        path_to_pickled_tokenizer = self.inputs["path_to_pickled_tokenizer"].read()
        path_to_scoring_data = self.inputs["path_to_scoring_data"].read()
        path_to_siamese_config = self.inputs["path_to_siamese_config"].read()

        path_to_result_folder = os.path.split(path_to_scoring_data)[0]
        test_df = pd.read_csv(path_to_scoring_data)
        test_df.fillna({'sentence1': '', 'sentence2': ''}, inplace=True)
        if "Unnamed: 0" in test_df.columns:
            del test_df["Unnamed: 0"]

        # filtering the test dataset based on grade range: +2/-2
        if filterby_grade_range != "nan":
            grade_range = 2
            test_df = filter_by_grade_range(test_df, grade_range)

        # filtering based on type of match
        if filterby_typeofmatch != "nan":
            test_df = test_df[test_df["type_of_match"] == filterby_typeofmatch].copy()
            test_df = test_df.reset_index(drop=True)
        print("****The best model path: ", path_to_best_model)

        # if model not present terminate the process:
        assert os.path.exists(path_to_result_folder)
        assert os.path.exists(path_to_best_model)

        # loading tokenizer:
        if not os.path.exists(path_to_pickled_tokenizer):
            print("Tokenizer not found")
        else:
            with open(path_to_pickled_tokenizer, 'rb') as tokenizer_file:
                tokenizer = pickle.load(tokenizer_file)

        with open(path_to_siamese_config, "rb") as json_file:
            siamese_config = json.load(json_file)

        # invoke the scoring module:
        output_pred_df = scoring_module(tokenizer, path_to_best_model, siamese_config, test_df, threshold)
        path_to_save_output = os.path.join(path_to_result_folder, "output_{0}.csv").format(embedding_method)
        output_pred_df.to_csv(path_to_save_output)
        self.outputs["path_to_predicted_output"].write(path_to_save_output)


class topic_level_aggregation(BaseOperator):
    @property
    def inputs(self):
        """
        Function that the TopicLevelAggregation operator defines while returning graph inputs

        :returns: Inputs to the node of the Content Reuse graph
            path_to_predicted_output: path to the output csv with predicted score
        """
        return {
            "path_to_predicted_output": File_Txt(self.node.inputs[0])
        }

    @property
    def outputs(self):
        """
        Function that the TopicLevelAggregation operator defines while returning graph outputs

        :returns: Returns the path to the csv with aggregated score for each topic pair
           path_to_output_topic_agg: path to the csv with aggregated score for each topic pair
        """
        return {"path_to_output_topic_agg": File_IO(
            self.node.outputs[0])}

    def run(self, aggregation_criteria, compute_topic_similarity, mandatory_column_names):
        path_to_predicted_output = self.inputs["path_to_predicted_output"].read()
        path_to_result_folder = os.path.split(path_to_predicted_output)[0]
        assert os.path.exists(path_to_predicted_output)
        output_pred_df = pd.read_csv(path_to_predicted_output)

        if "Unnamed: 0" in output_pred_df.columns:
            del output_pred_df["Unnamed: 0"]

        # Topic similarity aggregation computation:
        if compute_topic_similarity:
            path_to_save_output = os.path.join(path_to_result_folder, "agg_topic_level_output.csv")
            output_aggregated_topic_level = aggregation_topic_level(output_pred_df, aggregation_criteria,
                                                                    mandatory_column_names)
            output_aggregated_topic_level.to_csv(path_to_save_output)
        else:
            path_to_save_output = path_to_predicted_output
            print("*****Topic similarity aggregation not computed")
        self.outputs["path_to_output_topic_agg"].write(path_to_save_output)


class content_reuse_evaluation(BaseOperator):
    @property
    def inputs(self):
        """
        Function that the ContentReuseEvaluation operator defines while returning graph inputs
        :return: Inputs to the node of the Content Reuse graph
            path_to_full_score_metrics_file: path to scored dataframe file
        """
        return {
            "path_to_output_topic_agg": File_IO(self.node.inputs[0])
        }

    @property
    def outputs(self):
        """
        Function that the ContentReuseEvaluation operator defines while returning graph outputs
        :return: Returns the path to the k_eval_metrics
        """
        return {
            "path_to_k_eval_metrics": File_IO(self.node.outputs[0])
        }

    def run(self, window):
        """
        Generate k evaluation metrics
        :param window: length of k eval metrics window
        :return: None
        """
        path_to_output_topic_agg = self.inputs["path_to_output_topic_agg"].read()
        path_to_result_folder = os.path.split(path_to_output_topic_agg)[0]
        full_score_df = pd.read_csv(path_to_output_topic_agg)
        k_result_df = k_topic_recommendation(full_score_df, window)
        k_1 = k_result_df["k=1"].mean()
        k_2 = k_result_df["k=2"].mean()
        k_3 = k_result_df["k=3"].mean()
        k_4 = k_result_df["k=4"].mean()
        k_5 = k_result_df["k=5"].mean()
        eval_dict = {"k=1": k_1, "k=2": k_2, "k=3": k_3, "k=4": k_4, "k=5": k_5}
        with open(os.path.join(path_to_result_folder, 'k_eval_metrics.json'), 'w') as f:
            json.dump(eval_dict, f)
        self.outputs["path_to_k_eval_metrics"].write(os.path.join(path_to_result_folder, 'k_eval_metrics.json'))


class RecommendKConceptsPerTopic(BaseOperator):
    @property
    def inputs(self):
        """
        Function that the ContentReuseEvaluation operator defines while returning graph inputs
        :return: Inputs to the node of the Content Reuse graph
            path_to_full_score_metrics_file: path to scored dataframe file
        """
        return {
            "path_to_output_topic_agg": File_IO(self.node.inputs[0])
        }

    @property
    def outputs(self):
        """
        Function that the ContentReuseEvaluation operator defines while returning graph outputs
        :return: Returns the path to the k_eval_metrics
        """
        return {
            "path_to_dtb_mapping_file": File_IO(self.node.outputs[0])
        }

    def run(self, window):
        """
        Generate k evaluation metrics
        :param window: length of k eval metrics window
        :return: None
        """
        path_to_output_topic_agg = self.inputs["path_to_output_topic_agg"].read()
        path_to_dtb_mapping_file = os.path.join(os.path.split(path_to_output_topic_agg)[0], 'dtb_mapping.json')
        df = pd.read_csv(path_to_output_topic_agg)[['stb_id', 'concept_id', 'pred_score']]
        df.sort_values(['stb_id', 'pred_score'], ascending=False, inplace=True)
        output = {}
        for id_ in df.stb_id.unique():
            temp = df[df['stb_id'] == id_].head(window).set_index('concept_id').drop('stb_id', axis=1)
            output[id_] = [{ind: row['pred_score']} for ind, row in temp.iterrows()]
        with open(path_to_dtb_mapping_file, 'w') as f:
            json.dump(output, f)
        self.outputs["path_to_dtb_mapping_file"].write(path_to_dtb_mapping_file)


class WriteRelationshipsToNeo4j(BaseOperator):
    @property
    def inputs(self):
        """
        Function that the WriteRelationshipsToNeo4j operator defines while returning graph inputs
        :return: Inputs to the node of the Content Reuse graph
            path_to_configuration_file: path to configuration file
            path_to_dtb_mapping_file: path to dtb mapping file
        """
        return {
            "path_to_configuration_file": File_IO(self.node.inputs[0]),
            "path_to_dtb_mapping_file": File_IO(self.node.inputs[1])
        }

    @property
    def outputs(self):
        """
        Function that the WriteRelationshipsToNeo4j operator defines while returning graph outputs
        :return: Outputs to the node of the Content Reuse graph
        """
        return None

    def run(self):
        """
        Create a connection to Graph DB and ingest DTB Mapping relations to it
        """
        path_to_credentials = self.inputs["path_to_configuration_file"].read()
        path_to_dtb_mapping = self.inputs["path_to_dtb_mapping_file"].read()
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(path_to_credentials)
        try:
            scheme = ast.literal_eval(config["graph"]["scheme"])
            host = ast.literal_eval(config["graph"]["host"])
            port = ast.literal_eval(config["graph"]["port"])
            user = ast.literal_eval(config["graph"]["user"])
            password = ast.literal_eval(config["graph"]["password"])
            max_connections = ast.literal_eval(config["graph"]["max_connections"])
            secure = ast.literal_eval(config["graph"]["secure"])
            start_node_label = ast.literal_eval(config["relationship"]["start_node_label"])
            end_node_label = ast.literal_eval(config["relationship"]["end_node_label"])
            relationship_label = ast.literal_eval(config["relationship"]["relationship_label"])
            relationship_properties = ast.literal_eval(config["relationship"]["relationship_properties"])
            graph = connect_to_graph(scheme, host, port, user, password, max_connections, secure)
            with open(path_to_dtb_mapping, 'r') as f:
                dtb_mapping = json.load(f)
            create_node_relationships(graph, dtb_mapping, start_node_label, end_node_label, relationship_label,
                                      relationship_properties)
        except KeyError as ke:
            logging.error("Key Error found", ke.args, ke.__str__())
