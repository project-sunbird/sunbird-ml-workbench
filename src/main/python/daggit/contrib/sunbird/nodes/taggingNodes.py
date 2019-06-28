import multiprocessing
import time
import os
import glob
import json
import requests
import logging
import shutil
import nltk
import pandas as pd
import numpy as np
import configparser
from functools import partial
from elasticsearch import Elasticsearch

from daggit.core.io.io import Pandas_Dataframe, File_Txt
from daggit.core.io.io import ReadDaggitTask_Folderpath
from daggit.core.io.io import KafkaDispatcher, KafkaCLI
from daggit.core.base.factory import BaseOperator
from daggit.core.io.files import save_obj, load_obj, findFiles
from daggit.core.oplib.misc import df_feature_check, identify_contentType
from daggit.core.oplib.nlp import jaccard_with_phrase
from daggit.core.oplib.misc import merge_json
from daggit.core.oplib.nlp import strip_word, get_words
from daggit.core.oplib.misc import dictionary_merge, get_sorted_list
from daggit.core.oplib.nlp import custom_listPreProc
from daggit.core.oplib.misc import CustomDateFormater, findDate

from daggit.contrib.sunbird.oplib.taggingUtils import multimodal_text_enrichment
from daggit.contrib.sunbird.oplib.taggingUtils import keyword_extraction_parallel
from daggit.contrib.sunbird.oplib.taggingUtils import get_level_keywords
from daggit.contrib.sunbird.oplib.taggingUtils import precision_from_dictionary
from daggit.contrib.sunbird.oplib.taggingUtils import agg_precision_from_dictionary

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import partial, reduce
from kafka import KafkaProducer, KafkaConsumer, KafkaClient


class ContentToTextRead(BaseOperator):
    """
    An operator that extracts text for Content id within a specified range, in a Content meta dataframe

    :param range_start: specifies the start index of the dataframe 
    :type range_start: int
    :param range_end: specifies the end index of the dataframe
    :type range_end: int
    :param num_of_processes: number of workers used to distribute the process
    :type num_of_processes: int
    :param content_type: Specifies  the type of the content and related parameters
        Options are: ``{ youtube | pdf | ecml }
    :type content_type: json
    """
    @property
    def inputs(self):
        """
        Function that the ContentToTextRead operator defines while returning graph inputs

        :returns: Inputs to the node of the Auto tagging graph
            DS_DATA_HOME: a localpath where the folders get created
            localpathTocontentMeta: path to content meta
            pathTocredentials: path to config file with credentials 

        """
        return {
                "DS_DATA_HOME": ReadDaggitTask_Folderpath(self.node.inputs[0]),
                "localpathTocontentMeta": ReadDaggitTask_Folderpath(self.node.inputs[1]),
                "pathTocredentials": ReadDaggitTask_Folderpath(self.node.inputs[2])
                }

    @property
    def outputs(self):
        """
        Function that the ContentToTextRead operator defines while returning graph outputs

        :returns: Returns the path to timestamp folder in which auto tagging results get generated

        """
        return {"timestamp_folder": File_Txt(
                self.node.outputs[0])}

    def run(
            self,
            range_start,
            range_end,
            num_of_processes,
            content_type):
        """
        This is the main method to derive when creating an operator. This takes in the parameters, 
        runs text enrichment pipline and writes back the path to the 
        timestamp folder with the content id and its enriched text to an h5 file that gets saved as an intermediate result  

        """
        DS_DATA_HOME = self.inputs["DS_DATA_HOME"].read_loc()
        pathTocredentials = self.inputs["pathTocredentials"].read_loc()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        path_to_timestamp_folder = os.path.join(DS_DATA_HOME, timestr)
        content_to_text_path = os.path.join(
            path_to_timestamp_folder, "content_to_text")
        # content dump:
        if not os.path.exists(content_to_text_path):
            os.makedirs(content_to_text_path)
            print("content_to_text: ", content_to_text_path)
        
        contentmeta_path = self.inputs["localpathTocontentMeta"].read_loc()
        # move the content meta to timestamp folder[destination folder]
        #for the time being experiment with copy: change it later.
        shutil.move(contentmeta_path, os.path.join(path_to_timestamp_folder, os.path.split(contentmeta_path)[1]))
        moved_contentmeta_path = os.path.join(path_to_timestamp_folder, os.path.split(contentmeta_path)[1])
        
        content_meta = pd.read_csv(moved_contentmeta_path)
        if "derived_contentType" not in list(content_meta.columns):
            content_meta["derived_contentType"] = np.nan
            for row_ind, artifact_url in enumerate(content_meta["artifactUrl"]):
                try:
                    content_meta["derived_contentType"][row_ind] = identify_contentType(artifact_url)
                except BaseException:
                    pass
        content_meta = content_meta[pd.notnull(content_meta['derived_contentType'])]
        content_meta.reset_index(inplace=True, drop=True)
        print(self.outputs["timestamp_folder"].location_specify())
        oldwd = os.getcwd()
        contentMeta_mandatory_fields = [
            'artifactUrl',
            'derived_contentType',
            'downloadUrl',
            'gradeLevel',
            'identifier',
            'language',
            'subject',
            'graph_id',
            'nodeType',
            'objectType',
            'node_id']
        assert df_feature_check(content_meta, contentMeta_mandatory_fields)

        logging.info("CTT_CONTENT_TO_TEXT_START")
        # read content meta:
        if content_meta.columns[0] == "0":
            content_meta = content_meta.drop("0", axis=1)

        # check for duplicates in the meta
        if list(content_meta[content_meta.duplicated(
                ['artifactUrl'], keep=False)]["artifactUrl"]) != []:
            content_meta.drop_duplicates(subset="artifactUrl", inplace=True)
            content_meta.reset_index(drop=True, inplace=True)

        # dropna from artifactUrl feature and reset the index:
        content_meta.dropna(subset=["artifactUrl"], inplace=True)
        content_meta.reset_index(drop=True, inplace=True)

        # time the run
        start = time.time()
        logging.info(
            'Contents detected in the content meta: ' + str(len(content_meta)))
        logging.info(
            "----Running Content_to_Text for contents from {0} to {1}:".format(
                range_start, range_end))
        logging.info("time started: {0}".format(start))
        # subset contentMeta:
        # content_meta = content_meta[content_meta["derived_contentType"].isin(
        #     subset_contentMeta_by.split(", "))]
        content_meta.reset_index(drop=True, inplace=True)
        if range_start == "START":
            range_start = 0
        if range_end == "END":
            range_end = len(content_meta)
        logging.info(
            "CTT_Config: content_meta from {0} to {1} created in: {2}".format(
                range_start, range_end, content_to_text_path))
        print("Number of processes: ", num_of_processes)

        status = False
        if os.path.exists(pathTocredentials):
            try:
                config = configparser.ConfigParser(allow_no_value=True)
                config.read(pathTocredentials)
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

        result = [
            multimodal_text_enrichment(
                i,
                timestr,
                content_meta,
                content_type,
                content_to_text_path,
                GOOGLE_APPLICATION_CREDENTIALS) for i in range(
                range_start,
                range_end)]
        print(result)
        os.chdir(oldwd)
        print("Current directory c2t: ", os.getcwd())
        print("timestamp_folder path:", path_to_timestamp_folder)
        self.outputs["timestamp_folder"].write(path_to_timestamp_folder)


class KeywordExtraction(BaseOperator):
    
    @property
    def inputs(self):
        return {"timestamp_folder": File_Txt(self.node.inputs[0]),
                "pathTocredentials": ReadDaggitTask_Folderpath(self.node.inputs[1]),
                "categoryLookup": ReadDaggitTask_Folderpath(self.node.inputs[2])
                }

    @property
    def outputs(self):
        return {"path_to_contentKeywords": File_Txt(self.node.outputs[0])
                }

    def run(self, extract_keywords, filter_criteria, update_corpus, filter_score_val, num_keywords):
        assert extract_keywords == "tagme" or extract_keywords == "text_token"
        assert filter_criteria == "none" or filter_criteria == "taxonomy" or filter_criteria == "dbpedia"
        pathTocredentials = self.inputs["pathTocredentials"].read_loc()
        print("***************pathTocredentials: ", pathTocredentials)
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(pathTocredentials)
        cache_cred=dict()
        cache_cred['host']=config["redis"]["host"]
        cache_cred['port']=config["redis"]["port"]
        cache_cred['password']=config["redis"]["password"]
        
        tagme_cred = dict()
        tagme_cred['gcube_token']=config['tagme credentials']['gcube_token']
        tagme_cred['postman_token']=config['tagme credentials']['postman_token']                        

        path_to_category_lookup = self.inputs["categoryLookup"].read_loc()
        timestamp_folder = self.inputs["timestamp_folder"].read()
        timestr = os.path.split(timestamp_folder)[1]
        print("****timestamp folder:", timestamp_folder)
        print("****categorylookup yaml:", path_to_category_lookup)
        content_to_text_path = os.path.join(timestamp_folder, "content_to_text")
        print("content_to_text path:", content_to_text_path)
        if not os.path.exists(content_to_text_path):
            logging.info("No such directory as: ", content_to_text_path)
        else:
            logging.info('------Transcripts to keywords extraction-----')
            pool = multiprocessing.Pool(processes=4)
            keywordExtraction_partial = partial(
               keyword_extraction_parallel,
               timestr=timestr,
               content_to_text_path=content_to_text_path,
               extract_keywords=extract_keywords,
               filter_criteria=filter_criteria,
               cache_cred=cache_cred,
               path_to_category_lookup=path_to_category_lookup,
               update_corpus=update_corpus,
               filter_score_val=filter_score_val,
               num_keywords=num_keywords,
               tagme_cred=tagme_cred)
            results = pool.map(
                keywordExtraction_partial, [
                    dir for dir in os.listdir(content_to_text_path)])
            print(results)
            print("path to content keywords:", max(glob.glob(
                os.path.join(timestamp_folder, 'content_to_text'))))
            c2t_path = os.path.join(timestamp_folder, 'content_to_text')
            self.outputs["path_to_contentKeywords"].write(max(glob.glob(
                c2t_path), key=os.path.getmtime))

            pool.close()
            pool.join()


class WriteToElasticSearch(BaseOperator):

    @property
    def inputs(self):
        return {"timestamp_folder": File_Txt(self.node.inputs[0])
                }

    def run(self):
        timestamp_folder = self.inputs["timestamp_folder"].read()
        timestr = os.path.split(timestamp_folder)[1]
        epoch_time = time.mktime(time.strptime(timestr, "%Y%m%d-%H%M%S"))
        es_request = requests.get('http://localhost:9200')
        content_to_textpath = os.path.join(timestamp_folder, "content_to_text")
        cid_name = [i for i in os.listdir(content_to_textpath) if i not in ['.DS_Store']]
        for cid in cid_name:
            merge_json_list = []
            json_file = findFiles(os.path.join(content_to_textpath, cid), ["json"])
            logging.info("json_files are: ", json_file)
            for file in json_file:
                if os.path.split(file)[1] in [
                        "ML_keyword_info.json", "ML_content_info.json"]:
                    merge_json_list.append(file)
            autotagging_json = merge_json(merge_json_list)
            autotagging_json.update({"ETS": epoch_time})
            elastic_search = Elasticsearch(
                [{'host': 'es', 'port': 9200}]) #change it to localhost
            if es_request.status_code == 200:
                elastic_search.index(
                    index="auto_tagging",
                    doc_type='content_id_info',
                    id=cid,
                    body=autotagging_json)

class WriteToKafkaTopic(BaseOperator):

    @property
    def inputs(self):
        return {"path_to_contentKeywords": File_Txt(self.node.inputs[0]),
                "pathTocredentials": ReadDaggitTask_Folderpath(self.node.inputs[1])
                }

    def run(self, write_to_kafkaTopic):
        path_to_contentKeywords = self.inputs["path_to_contentKeywords"].read()
        pathTocredentials = self.inputs["pathTocredentials"].read_loc()
        timestamp_folder = os.path.split(path_to_contentKeywords)[0]
        timestr = os.path.split(timestamp_folder)[1]
        epoch_time = time.mktime(time.strptime(timestr, "%Y%m%d-%H%M%S"))
        content_to_textpath = os.path.join(timestamp_folder, "content_to_text")
        cid_name = [i for i in os.listdir(content_to_textpath) if i not in ['.DS_Store']]
        for cid in cid_name:
            merge_json_list = []
            json_file = findFiles(os.path.join(content_to_textpath, cid), ["json"])
            for file in json_file:
                if os.path.split(file)[1] in [
                        "ML_keyword_info.json", "ML_content_info.json"]:
                    merge_json_list.append(file)
            ignore_list = ["ets"]
            dict_list = []
            for file in merge_json_list:
                with open(file, "r", encoding="UTF-8") as info:
                    new_json = json.load(info)
                    [new_json.pop(ignore) for ignore in ignore_list if ignore in new_json.keys()]
                dict_list.append(new_json)
            # merge the nested jsons:-
            autotagging_json = reduce(merge_json, dict_list)
            autotagging_json.update({"ets": epoch_time})
            with open(os.path.join(timestamp_folder, "content_to_text", cid, "autoTagging_json.json"), "w+") as main_json:
                json.dump(autotagging_json, main_json, sort_keys=True, indent=4)
            # writing to kafka topic:-
            kafka_cli = KafkaCLI(pathTocredentials)
            status = kafka_cli.write(autotagging_json, write_to_kafkaTopic)
            if status:
                logging.info("******Transaction event successfully pushed to topic:{0}".format(write_to_kafkaTopic))

            else:
                logging.info("******Error pushing the event")
        # Remove the timestamp folder:-
        shutil.rmtree(timestamp_folder)

            
class CorpusCreation(BaseOperator):

    @property
    def inputs(self):
        return {"pathTotaxonomy": Pandas_Dataframe(self.node.inputs[0]),
                "path_to_contentKeywords": File_Txt(self.node.inputs[1])
                }

    @property  # how to write to a folder?
    def outputs(self):
        return {"root_path": File_Txt(self.node.outputs[0]),
                "path_to_corpus": File_Txt(self.node.outputs[1])
                }

    def run(self, keyword_folder_name, update_corpus, word_preprocess):
        keyword_folder_ls = ["tagme_none", "txt_token_none", "tagme_taxonomy", "tagme_dbpedia"]
        if keyword_folder_name in keyword_folder_ls:
            taxonomy = self.inputs["pathTotaxonomy"].read()
            path_to_contentKeys = self.inputs["path_to_contentKeywords"].read()
            keyword_folder = os.path.split(path_to_contentKeys)[0]
            corpus_creation_folder = os.path.join(keyword_folder, "corpus_creation")
            if not os.path.exists(corpus_creation_folder):
                os.makedirs(corpus_creation_folder)
            root_path = os.path.split(os.path.split(path_to_contentKeys)[0])[0]
            corpus_loc = os.path.join(root_path, "corpus")
            if not os.path.exists(corpus_loc):
                os.makedirs(corpus_loc)
            corpus_csv_loc = os.path.join(corpus_loc, "corpus.csv")
            vocabulary_loc = os.path.join(corpus_creation_folder, "vocab")
            cids = os.listdir(path_to_contentKeys)
            content_keywords_list = []
            for content in cids:
                path_to_keywords = os.path.join(
                    path_to_contentKeys,
                    content,
                    "keywords",
                    keyword_folder_name,
                    "keywords.csv")
                if not os.path.exists(path_to_keywords):
                    extracted_keys = []
                else:
                    extracted_keyword_df = pd.read_csv(
                        path_to_keywords, keep_default_na=False)
                    extracted_keys = list(extracted_keyword_df['keyword'])
                content_keywords_list.append(extracted_keys)
                # print("content_keywords_list: ", content_keywords_list)
            content_keywords_list = custom_listPreProc(
                content_keywords_list,
                word_preprocess["method"],
                word_preprocess["delimitter"])
            taxonomy['Keywords'] = [
                get_words(i) for i in list(
                    taxonomy['Keywords'])]
            taxonomy_keywords = [
                x for x in list(
                    taxonomy['Keywords']) if str(x) != 'nan']
            taxonomy_keywords = custom_listPreProc(
                taxonomy_keywords,
                word_preprocess["method"],
                word_preprocess["delimitter"])
            if os.path.exists(corpus_csv_loc):
                corpus = list(pd.read_csv(corpus_csv_loc)['Words'])
            else:
                corpus = []
            all_words = list(set(
                [i for item1 in taxonomy_keywords for i in item1] +
                [j for item2 in content_keywords_list for j in item2] +
                corpus))
            print("number of unique words: " + str(len(set(all_words))))
            vocabulary = dict()
            for i in range(len(all_words)):
                vocabulary[all_words[i]] = i
            save_obj(vocabulary, vocabulary_loc)
            if update_corpus:
                pd.DataFrame({'Words': all_words}).to_csv(corpus_csv_loc)
            self.outputs["root_path"].write(
                os.path.split(path_to_contentKeys)[0])
            self.outputs["path_to_corpus"].write(corpus_creation_folder)
        else:
            logging.info(" {0} is unknown name".format("keyword_folder_name"))


class ContentTaxonomyScoring(BaseOperator):

    @property
    def inputs(self):
        return {"localpathTocontentMeta": ReadDaggitTask_Folderpath(self.node.inputs[0]),
                "pathTotaxonomy": Pandas_Dataframe(self.node.inputs[1]),
                "root_path": File_Txt(self.node.inputs[2]),
                "path_to_corpus": File_Txt(self.node.inputs[3])

                }

    @property
    def outputs(self):
        return {"path_to_timestampFolder": File_Txt(self.node.outputs[0]),
                "path_to_distMeasure": File_Txt(self.node.outputs[1]),
                "path_to_domain_level": File_Txt(self.node.outputs[2])
                }

    def run(
            self,
            keyword_extract_filter_by,
            phrase_split,
            min_words,
            distanceMeasure,
            embedding_method,
            delimitter,
            filter_by):
        contentmeta_filterby_column = filter_by["contentMeta"]["column"]
        contentmeta_level = filter_by["contentMeta"]["alignment_depth"]
        taxonomy_filterby_column = filter_by["taxonomy"]["column"]
        taxonomy_level = filter_by["taxonomy"]["alignment_depth"]
        content_meta_loc = self.inputs["localpathTocontentMeta"].read_loc()
        taxonomy = self.inputs["pathTotaxonomy"].read()
        root_path = self.inputs["root_path"].read()
        corpus_folder = self.inputs["path_to_corpus"].read()
        content_meta = pd.read_csv(content_meta_loc)
        # check for the presence of corpus folder:
        if not os.path.exists(corpus_folder):
            logging.info("No corpus folder created")
        else:
            vocab_loc = corpus_folder + "/vocab"
            vocabulary = load_obj(vocab_loc)
        mapping_folder = os.path.join(root_path, "content_taxonomy_scoring")

        if not os.path.exists(mapping_folder):
            os.makedirs(mapping_folder)
        print("***mapping folder:", mapping_folder)

        if len(os.listdir(mapping_folder)) == 0:
            output = os.path.join(mapping_folder, "Run_0")
            os.makedirs(output)
        else:
            path_to_subfolders = [
                os.path.join(
                    mapping_folder,
                    f) for f in os.listdir(mapping_folder) if os.path.exists(
                    os.path.join(
                        mapping_folder,
                        f))]
            create_output = [
                os.path.join(
                    mapping_folder,
                    "Run_{0}".format(
                        i + 1)) for i,
                _ in enumerate(path_to_subfolders)]
            os.makedirs(create_output[-1])
            output = create_output[-1]
        print("***output:", output)
        DELIMITTER = delimitter
        # cleaning taxonomy KEYWORDS
        taxonomy['Keywords'] = [get_words(item) for item in list(
            taxonomy['Keywords'])]  # get words from string of words
        taxonomy_keywords = [
            x for x in list(
                taxonomy['Keywords']) if str(x) != 'nan']
        taxonomy_keywords = custom_listPreProc(
            taxonomy_keywords, 'stem_lem', DELIMITTER)

        # print("****Taxonomy_df keywords****: ", taxonomy["Keywords"])
        logging.info('Number of Content detected:  ' + str(len(content_meta)))
        print("Number of content detected:", str(len(content_meta)))

        content_keywords_list = []

        logging.info("******Content keyword creation for content meta*******")
        path_to_corpus = root_path + "/content_to_text"
        print("***path_to_corpus: ", path_to_corpus)
        if not os.path.exists(path_to_corpus):
            print("No such directory as path_to_corpus:", path_to_corpus)
        else:
            print(
                "list of folders in path_to_corpus: ",
                os.listdir(path_to_corpus))
            for content in content_meta['identifier']:
                if not os.path.exists(
                    os.path.join(
                        path_to_corpus,
                        content,
                        "keywords",
                        keyword_extract_filter_by,
                        "keywords.csv")):
                    extracted_keys = []
                else:
                    extracted_keyword_df = pd.read_csv(
                        os.path.join(
                            path_to_corpus,
                            content,
                            "keywords",
                            keyword_extract_filter_by,
                            "keywords.csv"),
                        keep_default_na=False)
                    print(
                        "keywords {0} for id {1}:".format(
                            list(
                                extracted_keyword_df['keyword']),
                            content))
                    extracted_keys = list(extracted_keyword_df['keyword'])
                content_keywords_list.append(extracted_keys)
            content_keywords_list = custom_listPreProc(
                content_keywords_list,
                'stem_lem',
                DELIMITTER)
            content_meta['Content_keywords'] = content_keywords_list
            content_meta = content_meta.iloc[[i for i, e in enumerate(
                content_meta['Content_keywords']) if (e != []) and len(e) > min_words]]
            content_meta = content_meta.reset_index(drop=True)
            print(
                "contentmeta domains:", set(
                    content_meta[contentmeta_filterby_column]))
            print("taxonomy domains:", set(taxonomy[taxonomy_filterby_column]))
            domains = list(set(
                content_meta[contentmeta_filterby_column]) & set(
                taxonomy[taxonomy_filterby_column]))
            print()
            print("Domains: ", domains)
            # empty domain
            if not domains:
                logging.info("No Subjects common")
            logging.info(
                "Aggregated on level: {0}".format(taxonomy_level))
            logging.info("------------------------------------------")
            content_meta_sub = content_meta[contentmeta_filterby_column]
            logging.info("***Skipping Content id: {0}".format(list(
                content_meta[~content_meta_sub.isin(domains)]['identifier'])))

            dist_all = dict()
            domain_level_all = dict()
            for i in domains:
                subject = [i]

                logging.info("Running for subject: {0}".format(subject))
                domain_content_df = content_meta.loc[content_meta_sub.isin(
                    subject)]  # filter arg: contentmeta column: subject
                domain_content_df.index = domain_content_df['identifier']
                tax_sub = taxonomy[taxonomy_filterby_column]
                domain_taxonomy_df = taxonomy.loc[tax_sub.isin(
                    subject)]  # filter arg: taxonomy column: Subject
                level_domain_taxonomy_df = get_level_keywords(
                    domain_taxonomy_df, taxonomy_level)
                if (distanceMeasure == 'jaccard' or distanceMeasure ==
                        'match_percentage') and embedding_method == "none":
                    level_domain_taxonomy_df.index = level_domain_taxonomy_df[taxonomy_level]

                    logging.info("Number of Content in domain: {0} ".format(
                        str(len(domain_content_df))))
                    logging.info("Number of Topic in domain: {0}".format(
                        str(len(level_domain_taxonomy_df))))
                    dist_df = pd.DataFrame(
                        np.zeros(
                            (len(domain_content_df),
                             len(level_domain_taxonomy_df))),
                        index=domain_content_df.index,
                        columns=level_domain_taxonomy_df.index)
                    if len(level_domain_taxonomy_df) > 1:
                        if phrase_split is True:
                            # rewrite the nested for loop:-(optimize the code)
                            for row_ind in range(dist_df.shape[0]):
                                for col_ind in range(dist_df.shape[1]):
                                    content_keywords = [strip_word(i, DELIMITTER) for i in domain_content_df['Content_keywords'][row_ind]]
                                    taxonomy_keywords = [strip_word(i, DELIMITTER) for i in level_domain_taxonomy_df['Keywords'][col_ind]]
                                    jaccard_index = jaccard_with_phrase(
                                        content_keywords, taxonomy_keywords)
                                    dist_df.iloc[row_ind,
                                                 col_ind] = jaccard_index[distanceMeasure]
                                    mapped_df = dist_df.T.apply(
                                        func=lambda x:
                                        get_sorted_list(x, 0), axis=0).T
                                    mapped_df.columns = range(
                                        1, mapped_df.shape[1] + 1)
                            domain_level_all['& '.join(subject)] = mapped_df
                            dist_all['& '.join(subject)] = dist_df

                if (distanceMeasure == 'cosine'):
                    if len(level_domain_taxonomy_df) > 1:
                        taxonomy_documents = [
                            " ".join(doc) for doc in list(
                                level_domain_taxonomy_df['Keywords'])]
                        content_documents = [
                            " ".join(doc) for doc in list(
                                domain_content_df['Content_keywords'])]
                        if embedding_method == 'tfidf':
                            vectorizer = TfidfVectorizer(vocabulary=vocabulary)
                        elif embedding_method == 'onehot':
                            vectorizer = CountVectorizer(vocabulary=vocabulary)
                        else:
                            print("unknown embedding_method")
                            print("selecting default sklearn.CountVectorizer")
                            vectorizer = CountVectorizer(vocabulary=vocabulary)
                        vectorizer.fit(list(vocabulary.keys()))
                        taxonomy_freq_df = vectorizer.transform(
                            taxonomy_documents)
                        taxonomy_freq_df = pd.DataFrame(
                            taxonomy_freq_df.todense(),
                            index=list(
                                level_domain_taxonomy_df[taxonomy_level]),
                            columns=vectorizer.get_feature_names())

                        content_freq_df = vectorizer.transform(
                            content_documents)
                        content_freq_df = pd.DataFrame(content_freq_df.todense(),
                                                       index=list(
                            domain_content_df.index),
                            columns=vectorizer.get_feature_names())
                        dist_df = pd.DataFrame(
                            cosine_similarity(
                                content_freq_df, taxonomy_freq_df), index=list(
                                domain_content_df.index), columns=list(
                                level_domain_taxonomy_df[taxonomy_level]))
                        mapped_df = dist_df.T.apply(
                            func=lambda x: get_sorted_list(x, 0), axis=0).T
                        mapped_df.columns = range(1, mapped_df.shape[1] + 1)
                        domain_level_all['& '.join(subject)] = mapped_df
                        dist_all['& '.join(subject)] = dist_df

            if not os.path.exists(output):
                os.makedirs(output)
            save_obj(dist_all, os.path.join(output, "dist_all"))
            save_obj(
                domain_level_all,
                os.path.join(
                    output,
                    "domain_level_all"))
            cts_output_dict = {
                'content_taxonomy_scoring': [
                    {
                        'distanceMeasure': distanceMeasure,
                        'Common domains for Taxonomy and ContentMeta': domains,
                        'keyword_extract_filter_by': keyword_extract_filter_by,
                        'embedding_method': embedding_method,
                        'filter_taxonomy': taxonomy_filterby_column,
                        'filter_meta': contentmeta_filterby_column,
                        'taxonomy_alignment_depth': taxonomy_level,
                        'content_meta_level': contentmeta_level,
                        'path_to_distanceMeasure': os.path.join(
                            output,
                            "dist_all.pkl"),
                        'path_to_domain_level': os.path.join(
                            output,
                            "domain_level_all.pkl")}]}

            with open(os.path.join(output, "ScoringInfo.json"), "w") as info:
                cts_json_dump = json.dump(
                    cts_output_dict,
                    info,
                    sort_keys=True,
                    indent=4)
            print(cts_json_dump)
            self.outputs["path_to_timestampFolder"].write(root_path)
            self.outputs["path_to_distMeasure"].write(
                os.path.join(output, "dist_all.pkl"))
            self.outputs["path_to_domain_level"].write(
                os.path.join(output, "domain_level_all.pkl"))


class PredictTag(BaseOperator):

    @property
    def inputs(self):
        return {"path_to_timestampFolder": File_Txt(self.node.inputs[0])
                }

    @property  # how to write to a folder?
    def outputs(self):
        return {"path_to_predictedTags": File_Txt(self.node.outputs[0])
                }

    def run(self, window):

        timestamp_folder = self.inputs["path_to_timestampFolder"].read()
        logging.info("PT_START")
        output = os.path.join(timestamp_folder, "content_taxonomy_scoring")
        print("output:", output)
        prediction_folder = os.path.join(timestamp_folder, "prediction")
        logging.info("PT_PRED_FOLDER_CREATED: {0}".format(prediction_folder))
        logging.info("PT_WINDOW: {0}". format(window))
        dist_dict_list = [
            load_obj(
                os.path.join(
                    output,
                    path_to_runFolder,
                    "domain_level_all")) for path_to_runFolder in os.listdir(output) if os.path.exists(
                os.path.join(
                    output,
                    path_to_runFolder,
                    "domain_level_all.pkl"))]
        dist_dict = dictionary_merge(dist_dict_list)
        print("dist_dict:", dist_dict)
        if bool(dist_dict) is False:
            logging.info("Dictionary list is empty. No tags to predict")
        else:
            if not os.path.exists(prediction_folder):
                os.makedirs(prediction_folder)
            pred_df = pd.DataFrame()
            for domain in dist_dict.keys():
                pred_df = pred_df.append(dist_dict[domain].iloc[:, 0:window])
            pred_df.to_csv(
                os.path.join(
                    prediction_folder,
                    "predicted_tags.csv"))
            self.outputs["path_to_predictedTags"].write(
                os.path.join(prediction_folder, "predicted_tags.csv"))
        logging.info("PT_END")


class GenerateObservedTag(BaseOperator):

    @property
    def inputs(self):
        return {"localpathTocontentMeta": ReadDaggitTask_Folderpath(self.node.inputs[0]),
                "pathTotaxonomy": Pandas_Dataframe(self.node.inputs[1]),
                "path_to_timestampFolder": File_Txt(self.node.inputs[2])
                }

    @property
    def outputs(self):
        return {"path_to_timestampFolder": File_Txt(self.node.outputs[0]),
                "path_to_observedtag": File_Txt(self.node.outputs[1]),
                "path_to_predictedtag": File_Txt(self.node.outputs[2])
                }

    def getGradedigits(self, class_x):
        for i in ["Class", "[", "]", " ", "class", "Grade", "grade"]:
            class_x = class_x.replace(i, "")
        return class_x

    def run(self, window, level, tax_known_tag, content_known_tag):

        content_meta_loc = self.inputs["localpathTocontentMeta"].read_loc()
        content_meta = pd.read_csv(content_meta_loc)
        content_meta = content_meta[pd.notnull(content_meta[content_known_tag])]
        taxonomy = self.inputs["pathTotaxonomy"].read()
        timestamp_folder = self.inputs["path_to_timestampFolder"].read()
        # mapping
        level_mapping = pd.Series(
            taxonomy[tax_known_tag].values, index=list(
                taxonomy[level])).to_dict()
        observed_col = pd.Series(
            content_meta[content_known_tag].values, index=list(
                content_meta['identifier'])).to_dict()
        # cleaning
        level_mapping = dict((k, self.getGradedigits(v))
                             for k, v in level_mapping.items())
        observed_col = dict((k, self.getGradedigits(v))
                            for k, v in observed_col.items())
        output = os.path.join(timestamp_folder, "content_taxonomy_scoring")
        if not os.path.exists(output):
            logging.info("Taxonomy mapping not performed")
        else:
            observed_tag_output = os.path.join(timestamp_folder, "generate_observed_tags")
            dist_dict_list = [
                load_obj(
                    os.path.join(
                        output,
                        path_to_runFolder,
                        "domain_level_all")) for path_to_runFolder in os.listdir(output) if os.path.exists(
                    os.path.join(
                        output,
                        path_to_runFolder,
                        "domain_level_all.pkl"))]
            predicted_tag = dictionary_merge(dist_dict_list)
            # check if dist_dict empty or not
            if bool(predicted_tag) is False:
                logging.info("No known-tag-discovery to be performed")
            else:
                if not os.path.exists(observed_tag_output):
                    os.makedirs(observed_tag_output)
                # tagging to known values
                observed_tag = dict()
                predicted_tag_known = dict()
                for domain in predicted_tag.keys():
                    cid = predicted_tag[domain].index
                    domain_obs_df = pd.DataFrame(
                        list(cid), index=cid, columns=[tax_known_tag])
                    domain_obs_df = domain_obs_df.replace(observed_col)
                    observed_tag[domain] = domain_obs_df
                    domain_pred_df = predicted_tag[domain].replace(
                        level_mapping)
                    predicted_tag_known[domain] = domain_pred_df
                save_obj(
                    observed_tag,
                    os.path.join(
                        observed_tag_output,
                        "observed_tags"))
                save_obj(
                    predicted_tag_known,
                    os.path.join(
                        observed_tag_output,
                        "predicted_tags"))
                self.outputs["path_to_timestampFolder"].write(
                    timestamp_folder)
                self.outputs["path_to_observedtag"].write(
                    os.path.join(observed_tag_output, "observed_tags.pkl"))
                self.outputs["path_to_predictedtag"].write(
                    os.path.join(observed_tag_output, "predicted_tags.pkl"))


class Evaluation(BaseOperator):

    @property
    def inputs(self):
        return {"path_to_timestampFolder": File_Txt(self.node.inputs[0]),
                "path_to_observedtag": File_Txt(self.node.inputs[1]),
                "path_to_predictedtag": File_Txt(self.node.inputs[2])
                }

    @property
    def outputs(self):
        return {"path_to_agg_precision": File_Txt(self.node.outputs[0]),
                "path_to_nonagg_precision": File_Txt(self.node.outputs[1])
                }

    def run(self, window):
        timestamp_folder = self.inputs["path_to_timestampFolder"].read()
        path_to_observedtag = self.inputs["path_to_observedtag"].read()
        path_to_predictedtag = self.inputs["path_to_predictedtag"].read()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        observed_tag = load_obj(path_to_observedtag[:-4])
        predicted_tag_known = load_obj(path_to_predictedtag[:-4])
        evaln_output = os.path.join(os.path.split(timestamp_folder)[
                                    0], "evaluation", timestr)  # use os.basepath()

        if bool(observed_tag) is False:
            logging.info("Observed tag is empty, No evaluation performed")
        else:
            if not os.path.exists(evaln_output):
                os.makedirs(evaln_output)
            eval_dct = dict()
            for items in observed_tag.keys():
                pred_df = predicted_tag_known[items]
                observed_df = observed_tag[items]
                eval_dct[items] = list(
                    precision_from_dictionary(
                        pred_df, observed_df, window)['percent'])
            pd.DataFrame(eval_dct).to_csv(os.path.join(
                evaln_output, "non_agg_precision.csv"))
            agg_prec_df = agg_precision_from_dictionary(
                predicted_tag_known, observed_tag, window)
            agg_prec_df.to_csv(os.path.join(evaln_output, "agg_precision.csv"))
            self.outputs["path_to_agg_precision"].write(
                os.path.join(evaln_output, "agg_precision.csv"))
            self.outputs["path_to_nonagg_precision"].write(
                os.path.join(evaln_output, "non_agg_precision.csv"))
