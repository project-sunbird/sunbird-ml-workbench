import multiprocessing
import time
import os
import glob
import logging
import pandas as pd
import numpy as np
from functools import partial

from daggit.contrib.ekstep.io.io import Pandas_Dataframe, Read_Folder, Pickle_Obj, File_Txt
from daggit.core.base.factory import BaseOperator
from ..operators.contentTaggingUtils import clean_url, identify_fileType, download_to_local,video_to_speech, multimodal_text_enrichment
from ..operators.contentTaggingUtils import speech_to_text, image_to_text, pdf_to_text, ecml_index_to_text
from ..operators.contentTaggingUtils import get_tagme_longtext, pafy_text_tokens, keyword_extraction_parallel
from ..operators.contentTaggingUtils import word_proc, get_words, clean_string_list ,stem_lem, get_level_keywords, jaccard_with_phrase, save_obj, load_obj
from ..operators.contentTaggingUtils import custom_listPreProc, dictionary_merge, get_sorted_list, get_prediction
from ..operators.contentTaggingUtils import getEvalMatrix, content_meta_features_checking

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentToText(BaseOperator):

    @property
    def inputs(self):
        return {"pathTocontentMeta": Pandas_Dataframe(self.node.inputs[0])
                }

    @property  # how to write to a folder?
    def outputs(self):
        return {"DS_DATA_HOME": Read_Folder(self.node.outputs[0]),
                "timestamp_folder": File_Txt(self.node.outputs[1])
                }
    
    
    def run(self, range_start, range_end, num_of_processes, subset_contentMeta_by, content_type):
        content_meta = self.inputs["pathTocontentMeta"].read()
        DS_DATA_HOME = self.outputs["DS_DATA_HOME"].read_loc()
        print("****DS_DATA_HOME: ", DS_DATA_HOME)
        print(self.outputs["timestamp_folder"].location_specify())
        oldwd = os.getcwd()
        contentMeta_mandatory_fields = ['artifactUrl', 'content_type','downloadUrl', 'gradeLevel', 'identifier','keywords', 'language', 'subject']
        assert content_meta_features_checking(content_meta, contentMeta_mandatory_fields) == True
        timestr = time.strftime("%Y%m%d-%H%M%S")
        content_to_text_path = os.path.join(DS_DATA_HOME, timestr, "content_to_text")
        
        # content dump:
        if not os.path.exists(content_to_text_path):
            os.makedirs(content_to_text_path)
            print("content_to_text: ", content_to_text_path)
       
        logging.info("CTT_CONTENT_TO_TEXT_START")
        logging.info(
            "CTT_Config: content_meta from range_start: {0} to range_end: {1} created in: {2}".format(
                range_start, range_end, content_to_text_path))

        # read content meta:
        if content_meta.columns[0] == "0":
            content_meta = content_meta.drop("0", axis=1)

        # check for duplicates in the meta
        if list(content_meta[content_meta.duplicated(['artifactUrl'], keep=False)]["artifactUrl"]) != []:
            content_meta.drop_duplicates(subset="artifactUrl", inplace=True)
            content_meta.reset_index(drop=True, inplace=True)

        # dropna from artifactUrl feature and reset the index:
        content_meta.dropna(subset=["artifactUrl"], inplace=True)
        content_meta.reset_index(drop=True, inplace=True)

        # time the run
        start = time.time()
        logging.info('Number of Content detected in the content meta: ' + str(len(content_meta)))
        logging.info(
            "-----Running Content to Text for contents from index {0} to index {1}:-----".format(range_start, range_end))
        logging.info("time started: {0}".format(start))
        
        #subset contentMeta:
        content_meta = content_meta[content_meta["content_type"].isin(subset_contentMeta_by.split(", "))]
        content_meta.reset_index(drop=True, inplace=True)
        
        print("Number of processes: ", num_of_processes)
        
        #pool = multiprocessing.Pool(processes=int(num_of_processes))
        result = [multimodal_text_enrichment(i, content_meta, content_type, content_to_text_path) for i in range(range_start, range_end)]
        # contentTotext_partial = partial(multimodal_text_enrichment, content_meta=content_meta,
        #                                 content_type=content_type, content_to_text_path=content_to_text_path)  # prod_x has only one argument x (y is fixed to 10)
        # results = pool.map(contentTotext_partial, [i for i in range(range_start, range_end)])
        # print(results)
        #changing working dir
        os.chdir(oldwd)
        print("Current directory c2t: ", os.getcwd())
        print("latest_folder_c2t:", max(glob.glob(os.path.join(DS_DATA_HOME, '*/')), key=os.path.getmtime))
        self.outputs["timestamp_folder"].write(max(glob.glob(os.path.join(DS_DATA_HOME, '*/')), key=os.path.getmtime))
        # pool.close()
        # pool.join()


class KeywordExtraction(BaseOperator):

    @property
    def inputs(self):
        return {"pathTocontentMeta": Pandas_Dataframe(self.node.inputs[0]),
                "pathTotaxonomy": Pandas_Dataframe(self.node.inputs[1]),           
                "timestamp_folder": File_Txt(self.node.inputs[2])

                }

    @property 
    def outputs(self):
        return {"path_to_contentKeywords": File_Txt(self.node.outputs[0]) 
                }

    def run(self, extract_keywords, filter_criteria):
        assert extract_keywords == "tagme" or extract_keywords == "text_token"
        assert filter_criteria == "none" or filter_criteria == "taxonomy" or filter_criteria == "dbpedia"
        content_meta = self.inputs["pathTocontentMeta"].read()
        taxonomy = self.inputs["pathTotaxonomy"].read()
        timestamp_folder = self.inputs["timestamp_folder"].read()
        print("****timestamp folder:", timestamp_folder)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        content_to_text_path = timestamp_folder+"content_to_text"
        print("content_to_text path:", content_to_text_path)
        if not os.path.exists(content_to_text_path):
            logging.info("No such directory as: ", content_to_text_path)
        
        else:  
            logging.info('------Transcripts to keywords extraction-----')
            
            pool = multiprocessing.Pool(processes=4)
            keywordExtraction_partial = partial(keyword_extraction_parallel,
                                                content_to_text_path=content_to_text_path, taxonomy=taxonomy, extract_keywords=extract_keywords, filter_criteria=filter_criteria)  # prod_x has only one argument x (y is fixed to 10)
            results = pool.map(keywordExtraction_partial, [dir for dir in os.listdir(content_to_text_path)])
            print("path to content keywords:", max(glob.glob(os.path.join(timestamp_folder[:-1], 'content_to_text'))))
            self.outputs["path_to_contentKeywords"].write(max(glob.glob(os.path.join(timestamp_folder[:-1], 'content_to_text')), key=os.path.getmtime))

            pool.close()
            pool.join()

class CorpusCreation(BaseOperator):

    @property
    def inputs(self):
        return { "pathTotaxonomy": Pandas_Dataframe(self.node.inputs[0]),
                 "path_to_contentKeywords": File_Txt(self.node.inputs[1])          
                }

    @property  # how to write to a folder?
    def outputs(self):
        return {"root_path": File_Txt(self.node.outputs[0]),
                "path_to_corpus": File_Txt(self.node.outputs[1])
                }

    def run(self, Keyword_folder_name, update_corpus, word_preprocess):
        assert Keyword_folder_name == "tagme_none" or Keyword_folder_name == "text_token_none" or Keyword_folder_name == "tagme_taxonomy"

        taxonomy = self.inputs["pathTotaxonomy"].read()
        path_to_contentKeywords = self.inputs["path_to_contentKeywords"].read()
        corpus_folder = os.path.split(path_to_contentKeywords)[0]+"/corpus"
        if not os.path.exists(corpus_folder):
            os.makedirs(corpus_folder)
        root_path = os.path.split(os.path.split(path_to_contentKeywords)[0])[0]
        corpus_loc=os.path.join(root_path,"corpus.csv")
        vocabulary_loc=os.path.join(corpus_folder,"vocab")
        cids=os.listdir(path_to_contentKeywords)
        
        content_keywords_list = []

        for content in cids:
            
            path_to_keywords=os.path.join(path_to_contentKeywords,content,"keywords", Keyword_folder_name, "keywords.csv")

            if not os.path.exists(path_to_keywords):
                extracted_keys = []
            else:
                extracted_keyword_df=pd.read_csv(path_to_keywords, keep_default_na=False)
                extracted_keys=list(extracted_keyword_df['KEYWORDS'])
            
            content_keywords_list.append(extracted_keys)
            print("content_keywords_list: ",content_keywords_list)

        content_keywords_list=custom_listPreProc(content_keywords_list, word_preprocess["method"], word_preprocess["delimitter"])

        taxonomy['Keywords']=[get_words(i) for i in list(taxonomy['Keywords'])]  #get words from string of words
        taxonomy_keywords=[x for x in list(taxonomy['Keywords']) if str(x) != 'nan']  # remove nan
        taxonomy_keywords=custom_listPreProc(taxonomy_keywords, word_preprocess["method"], word_preprocess["delimitter"])

        if os.path.exists(corpus_loc):
            corpus=list(pd.read_csv(corpus_loc)['Words'])
        else:
            corpus=[]
        all_words=list(set([i for item1 in taxonomy_keywords for i in item1] + [j for item2 in content_keywords_list for j in item2] + corpus))
        print(all_words)
        print("number of unique words: "+str(len(set(all_words))))


        vocabulary=dict()
        for i in range(len(all_words)):
            vocabulary[all_words[i]]=i
            
        save_obj(vocabulary,vocabulary_loc)
        if update_corpus==True:
            pd.DataFrame({'Words':all_words}).to_csv(corpus_loc) 
        self.outputs["root_path"].write(os.path.split(path_to_contentKeywords)[0])         
        self.outputs["path_to_corpus"].write(corpus_folder)

class ContentTaxonomyScoring(BaseOperator):

    @property
    def inputs(self):
        return {"pathTocontentMeta": Pandas_Dataframe(self.node.inputs[0]),
                "pathTotaxonomy": Pandas_Dataframe(self.node.inputs[1]),
                "root_path": File_Txt(self.node.inputs[2]),
                "path_to_corpus": File_Txt(self.node.inputs[3])

                }

    @property  # how to write to a folder?
    def outputs(self):
        return {"path_to_timestampFolder": File_Txt(self.node.outputs[0]),
                "path_to_distMeasure": File_Txt(self.node.outputs[1])
                }

    def run(self, keyword_extract_filter_by, phrase_split, distanceMeasure, embedding_method, delimitter, filter_by):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        assert keyword_extract_filter_by == "tagme_none" or keyword_extract_filter_by == "text_token_none" or keyword_extract_filter_by == "tagme_taxonomy"
        
        #arguments
        contentmeta_filterby_column = filter_by["contentMeta"]["column"] #subject
        contentmeta_alignmentDepth = filter_by["taxonomy"]["alignment_depth"] #none
        taxonomy_filterby_column = filter_by["taxonomy"]["column"] #Subject
        taxonomy_alignmentDepth = filter_by["taxonomy"]["alignment_depth"]#Chapter Name
        
        content_meta = self.inputs["pathTocontentMeta"].read()
        taxonomy = self.inputs["pathTotaxonomy"].read()
        root_path = self.inputs["root_path"].read()
        corpus_folder = self.inputs["path_to_corpus"].read()
        #check for the presence of corpus folder:
        if not os.path.exists(corpus_folder):
           logging.info("No corpus folder created")
        else:
            vocab_loc = corpus_folder+"/vocab"
            vocabulary=load_obj(vocab_loc)
        

        mapping_folder = root_path+"/content_taxonomy_scoring"

        if not os.path.exists(mapping_folder):
            os.makedirs(mapping_folder)
        print("***mapping folder:", mapping_folder)

        if len(os.listdir(mapping_folder)) == 0:
            output = os.path.join(mapping_folder, "Run_0")
            os.makedirs(output)
        else:
            path_to_subfolders = [os.path.join(mapping_folder, f) for f in os.listdir(mapping_folder) if os.path.exists(os.path.join(mapping_folder, f))]
            create_output = [os.path.join(mapping_folder, "Run_{0}".format(i+1)) for i, _ in enumerate(path_to_subfolders)]
            os.makedirs(create_output[-1])
            output = create_output[-1]
        print("***output:", output)


        DELIMITTER = delimitter
        # cleaning taxonomy KEYWORDS
        taxonomy['Keywords']=[get_words(item) for item in list(taxonomy['Keywords'])]  #get words from string of words
        taxonomy_keywords=[x for x in list(taxonomy['Keywords']) if str(x) != 'nan']  
        taxonomy_keywords=custom_listPreProc(taxonomy_keywords,'stem_lem',DELIMITTER) 

        print("****Taxonomy_df keywords****: ", taxonomy["Keywords"])

        logging.info('Number of Content detected:  ' +str(len(content_meta)))
        print("Number of content detected:", str(len(content_meta)))

        content_keywords_list = []

        logging.info("******Content keyword creation for content meta*******")
        path_to_corpus = root_path+"/content_to_text"
        print("***path_to_corpus: ", path_to_corpus)
    
        if not os.path.exists(path_to_corpus):
            print("No such directory as path_to_corpus:", path_to_corpus)
        else:
            print("list of folders in path_to_corpus: ", os.listdir(path_to_corpus))
            for content in content_meta['identifier']:
                #logging.info("content keywordlist creation for id:{0}".format(content))
                if not os.path.exists(os.path.join(path_to_corpus ,content, "keywords", keyword_extract_filter_by, "keywords.csv")):
                    extracted_keys = []
                else:
                    extracted_keyword_df =pd.read_csv(os.path.join(path_to_corpus ,content, "keywords", keyword_extract_filter_by, "keywords.csv"), keep_default_na=False)
                    print("keywords {0} for id {1}:".format(list(extracted_keyword_df['KEYWORDS']), content))
                    extracted_keys =list(extracted_keyword_df['KEYWORDS'])
                content_keywords_list.append(extracted_keys)
            

            content_keywords_list=custom_listPreProc(content_keywords_list,'stem_lem', DELIMITTER) #??word preprocess method==stem_lem
            content_meta['Content_keywords'] =content_keywords_list
            content_meta = content_meta.iloc[[i for i ,e in enumerate(content_meta['Content_keywords']) if (e!=[] and len(e)>5)]]
            content_meta = content_meta.reset_index(drop=True)

            # specify the level at which the aggregation needs to take place
            print("contentmeta domains:", set(content_meta[contentmeta_filterby_column]))
            print("taxonomy domains:", set(taxonomy[taxonomy_filterby_column]))
            domains = list(set(content_meta[contentmeta_filterby_column]) & set(taxonomy[taxonomy_filterby_column]))
            print()
            print("content meta columns: ", content_meta.columns)
            print("taxonomy columns:", taxonomy.columns)
            print("Domains: ", domains)

            if not domains:#empty domain
                logging.info("No Subjects common")
            logging.info("Aggregated on level: {0}".format(taxonomy_alignmentDepth))
            logging.info("------------------------------------------")

            logging.info("***Skipping Content id: {0}".format(list(content_meta[~content_meta[contentmeta_filterby_column].isin(domains)]['identifier'])))

            dist_all =dict()
            for i in domains:
                subject = [i]

                logging.info("Running for subject: {0}".format(subject))
                domain_content_df =content_meta.loc[content_meta[contentmeta_filterby_column].isin(subject)] #filter arg: contentmeta column: subject
                domain_content_df.index =domain_content_df['identifier']
                domain_taxonomy_df =taxonomy.loc[taxonomy[taxonomy_filterby_column].isin(subject)]#filter arg: taxonomy column: Subject
                level_domain_taxonomy_df =get_level_keywords(domain_taxonomy_df ,taxonomy_alignmentDepth) #level is "alignment_depth"
                if (distanceMeasure=='jaccard' or distanceMeasure=='match_percentage') and embedding_method == "none":
                    print("****", level_domain_taxonomy_df) 
                    level_domain_taxonomy_df.index =level_domain_taxonomy_df[taxonomy_alignmentDepth]

                    logging.info("Number of Content in domain: {0} ".format(str(len(domain_content_df)))) #??
                    logging.info("Number of Topic in domain: {0}".format(str(len(level_domain_taxonomy_df))))#??
                    dist_df =pd.DataFrame(np.zeros((len(domain_content_df) ,len(level_domain_taxonomy_df)))#??
                                                   ,index=domain_content_df.index ,columns=level_domain_taxonomy_df.index)#??

                    if len(level_domain_taxonomy_df) > 1: 
                        if phrase_split == True:
                            for row_ind in range(dist_df.shape[0]):
                                for col_ind in range(dist_df.shape[1]):
                                    content_keywords =list(map(word_proc, domain_content_df['Content_keywords'][row_ind]))#??
                                    taxonomy_keywords =list(map(word_proc,level_domain_taxonomy_df['Keywords'][col_ind]))#??
                                    jaccard_index = jaccard_with_phrase(content_keywords ,taxonomy_keywords)#??
                                    dist_df.iloc[row_ind ,col_ind ] =jaccard_index[distanceMeasure]#??
                                    mapped_df=dist_df.T.apply(func=lambda x:get_sorted_list(x,0),axis=0).T
                                    mapped_df.columns=range(1,mapped_df.shape[1]+1)
                            dist_all['& '.join(subject)] = mapped_df 

                if (distanceMeasure=='cosine'): 
                    if len(level_domain_taxonomy_df) > 1:
                        taxonomy_documents = [" ".join(doc) for doc in list(level_domain_taxonomy_df['Keywords'])]#??
                        content_documents = [" ".join(doc) for doc in list(domain_content_df['Content_keywords'])]#??
                        if embedding_method=='tfidf':
                            vectorizer = TfidfVectorizer(vocabulary = vocabulary)
                        elif embedding_method=='onehot':
                            vectorizer = CountVectorizer(vocabulary = vocabulary)
                        else:
                            print("unknown embedding_method")
                            print("selecting default method sklearn.CountVectorizer")
                            vectorizer = CountVectorizer(vocabulary = vocabulary)
                        vectorizer.fit(list(vocabulary.keys()))
                        taxonomy_freq_df=vectorizer.transform(taxonomy_documents)
                        taxonomy_freq_df=pd.DataFrame(taxonomy_freq_df.todense(), index=list(level_domain_taxonomy_df[taxonomy_alignmentDepth]), columns=vectorizer.get_feature_names())

                        content_freq_df=vectorizer.transform(content_documents)
                        content_freq_df=pd.DataFrame(content_freq_df.todense(), index=list(domain_content_df.index), columns=vectorizer.get_feature_names())
                        dist_df=pd.DataFrame(cosine_similarity(content_freq_df,taxonomy_freq_df), index=list(domain_content_df.index), columns=list(level_domain_taxonomy_df[level]))
                        mapped_df=dist_df.T.apply(func=lambda x:get_sorted_list(x,0),axis=0).T
                        mapped_df.columns=range(1,mapped_df.shape[1]+1)
                        dist_all['& '.join(subject)] = mapped_df 

            if not os.path.exists(output):
                os.makedirs(output)
            save_obj(dist_all, os.path.join(output,"dist_all"))
            with open(os.path.join(output, "ContentTaxonomyScoringInfo.txt"), "w") as contenttaxonomyscoring_info:
                contenttaxonomyscoring_info.writelines("DistanceMeasure: {0} \n".format(distanceMeasure))
                contenttaxonomyscoring_info.writelines("Common domains for Taxonomy and ContentMeta: {0} \n".format(domains))
                contenttaxonomyscoring_info.writelines("Folder with keywords: {0} \n".format(keyword_extract_filter_by))
            self.outputs["path_to_timestampFolder"].write(root_path)
            self.outputs["path_to_distMeasure"].write(os.path.join(output,"dist_all"))

class PredictTag(BaseOperator):

    @property
    def inputs(self):
        return {"path_to_timestampFolder": File_Txt(self.node.inputs[0])
                }

    @property  # how to write to a folder?
    def outputs(self):
        return {"path_to_timestampFolder1": File_Txt(self.node.outputs[0]),
                "path_to_predictedTags": File_Txt(self.node.outputs[1])
                }

    def run(self, window):

        level_pred_dict=dict()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        timestamp_folder = self.inputs["path_to_timestampFolder"].read()
        logging.info("PT_START")
        output = timestamp_folder+"/content_taxonomy_scoring"
        print("output:", output)
        prediction_folder = timestamp_folder+"/prediction"
        
        logging.info("PT_PRED_FOLDER_CREATED: {0}".format(prediction_folder))
        logging.info("PT_WINDOW: {0}". format(window))
        dist_dict_list=[load_obj(os.path.join(output, path_to_runFolder, "dist_all")) for path_to_runFolder in os.listdir(output) if os.path.exists(os.path.join(output, path_to_runFolder, "dist_all.pkl"))]
        dist_dict = dictionary_merge(dist_dict_list)
        if bool(dist_dict) == False:
            logging.info("Dictionary list is empty. No tags to pedict")
        else:
            if not os.path.exists(prediction_folder):
                os.makedirs(prediction_folder)
            level_pred_df = pd.DataFrame()
            for subject in dist_dict.keys():
                level_pred_df = pd.concat([dist_dict[subject], level_pred_df])
            pred_df=level_pref_df.iloc[:,0:window]
            pred_df.to_csv(os.path.join(prediction_folder,"predicted_tags.csv"))
            self.outputs["path_to_timestampFolder1"].write(timestamp_folder)
            self.outputs["path_to_predictedTags"].write(os.path.join(prediction_folder,"predicted_tags.csv"))
        
        # dist_dict = dictionary_merge(dist_dict_list)
        # print(dist_dict_list)
        # for subject in dist_dict.keys():
        #     logging.info("Subject detected in distance dictionary: {0}".format(subject))
        #     level_pred_dict[subject]=(get_prediction(dist_dict[subject],sort_order,window))
        # save_obj(level_pred_dict, os.path.join(prediction_folder,"predicted_tags"))
        logging.info("PT_END")

class KnownTagsDiscovery(BaseOperator):

    @property
    def inputs(self):
        return {"pathTocontentMeta": Pandas_Dataframe(self.node.inputs[0]),
                "pathTotaxonomy": Pandas_Dataframe(self.node.inputs[1]),
                "path_to_timestampFolder1": File_Txt(self.node.inputs[2])
                }

    @property  
    def outputs(self):
        return {"path_to_EvaluationMatrix": File_Txt(self.node.outputs[0])
                }

    def getGradedigits(self, class_x):
        for i in ["Class","[","]"," ","class","Grade","grade"]:
            class_x=class_x.replace(i,"")
        return class_x

    def run(self, window, level, tax_known_tag, content_known_tag):

        level_pred_dict=dict()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        content_meta = self.inputs["pathTocontentMeta"].read()
        taxonomy = self.inputs["pathTotaxonomy"].read()
        timestamp_folder = self.inputs["path_to_timestampFolder1"].read()
        #mapping
        level_mapping=pd.Series(taxonomy_df[tax_known_tag].values,index=list(taxonomy_df[level])).to_dict()
        observed_col=pd.Series(content_meta[content_known_tag].values,index=list(content_meta['identifier'])).to_dict()
        #cleaning
        level_mapping=dict((k, self.getGradedigits(v)) for k, v in level_mapping.items())
        observed_col=dict((k, self.getGradedigits(v)) for k, v in observed_col.items())
        
        output=timestamp_folder+"/content_taxonomy_scoring"
        if not os.path.exists(output):
            logging.info("Taxonomy mapping not performed")
        else:
            evaluation_output = os.path.join(os.path.split(timestamp_folder)[0]+"/evaluation",timestr)
            
            dist_dict_list=[load_obj(os.path.join(output, path_to_runFolder,"dist_all")) for path_to_runFolder in os.listdir(output) if os.path.exists(os.path.join(output, path_to_runFolder, "dist_all.pkl"))]
            dist_dict = dictionary_merge(dist_dict_list)
            if bool(dist_dict) == False:
                logging.info("No known-tag-discovery to be performed")
            else:   
                if not os.path.exists(evaluation_output):
                    os.makedirs(evaluation_output)
                    ## tagging to known values
                observed_tag=dict()
                predicted_tag_known=dict()
                for domain in dist_dict.keys():
                    cid=dist_dict[domain].index
                    domain_obs_df=pd.DataFrame(list(cid),index=cid, columns=[tax_known_tag])
                    domain_obs_df.replace(observed_col)
                    observed_tag[domain]=domain_obs_df
                    
                    domain_pred_df=dist_dict[domain].replace(unknown_known_mapping)
                    predicted_tag_known[domain]=domain_pred_df

                save_obj(observed_tag, os.path.join(evaluation_output, "observed_tags"))
                save_obj(predicted_tag_known, os.path.join(evaluation_output, "predicted_tags"))

