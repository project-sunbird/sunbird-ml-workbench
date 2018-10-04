
import multiprocessing
import time
import os
import glob
import logging
import pandas as pd
import numpy as np
from functools import partial

from daggit.iolib.io import Pandas_Dataframe, Read_Folder, Pickle_Obj, File_Txt
from daggit.core.base import BaseOperator
from ..operators_registry import get_op_callable
from ..core.contentTaggingUtils import clean_url, identify_fileType, content_to_text_conversion, multimode_text_enrichment
from ..core.contentTaggingUtils import get_tagme_longtext, pafy_text_tokens
from ..core.contentTaggingUtils import word_proc, get_words, clean_string_list ,stem_lem, get_level_keywords, jaccard_with_phrase, save_obj, load_obj
from ..core.contentTaggingUtils import custom_listPreProc, dictionary_merge, get_sorted_list, get_prediction, getGradedigits

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentToText(BaseOperator):

    @property
    def inputs(self):
        return {"content_meta": Pandas_Dataframe(self.node.inputs[0])
                }

    @property  # how to write to a folder?
    def outputs(self):
        return {"DS_DATA_HOME": Read_Folder(self.node.outputs[0]),
                "timestamp_folder": File_Txt(self.node.outputs[1])
                }

    def run(self, range_start, range_end, num_of_processes, url_type, youtube_extraction, ecml, pdf):
        content_meta = self.inputs["content_meta"].read()
        DS_DATA_HOME = self.outputs["DS_DATA_HOME"].read_loc()
        print("****DS_DATA_HOME", DS_DATA_HOME)
        print(self.outputs["timestamp_folder"].location_specify())

        timestr = time.strftime("%Y%m%d-%H%M%S")
        content_to_text_path = os.path.join(DS_DATA_HOME, timestr, "content_to_text")
        
        # log_file:
        # log_live = self.outputs["path_to_LogFile"].read_loc()
        # log_file = os.path.join(log_live, "content_to_text", timestr)
        # if not os.path.exists(log_file):
        #     os.makedirs(log_file)

        # content dump:
        if not os.path.exists(content_to_text_path):
            os.makedirs(content_to_text_path)
            print("content_to_text: ", content_to_text_path)

        #setting up logging
        # logging.basicConfig(filename=os.path.join(log_file, "content_to_text.log"), filemode='w', level=logging.DEBUG,
        #                     format='%(asctime)s %(message)s')

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

        # dropna from File Path feature and reset the index:
        content_meta.dropna(subset=["artifactUrl"], inplace=True)
        content_meta.reset_index(drop=True, inplace=True)

        # time the run
        start = time.time()
        logging.info('Number of Content detected in the content meta: ' + str(len(content_meta)))
        logging.info(
            "-----Running Content to Text for contents from index {0} to index {1}:-----".format(range_start, range_end))
        logging.info("time started: {0}".format(start))

        # check for youtube url type:-
        if url_type == "youtube":
            downloadField = youtube_extraction['contentDownloadField']

        elif url_type == "ecml":
            downloadField = ecml['contentDownloadField']

        else:
            downloadField = pdf['contentDownloadField']

        content_meta = content_meta[content_meta["content_type"] == url_type]
        content_meta.reset_index(drop=True, inplace=True)

        print("DownloadField: ", downloadField)
        print("Number of processes: ", num_of_processes)
        
        print("Parallelising!!!!!")
        # if __name__ == "__main__":
        pool = multiprocessing.Pool(processes=int(num_of_processes))
        contentTotext_partial = partial(multimode_text_enrichment, content_meta=content_meta,
                                        downloadField=downloadField, content_to_text_path=content_to_text_path)  # prod_x has only one argument x (y is fixed to 10)
        results = pool.map(contentTotext_partial, [i for i in range(range_start, range_end)])
        print(results)
        print("latest_folder_c2t:", max(glob.glob(os.path.join(DS_DATA_HOME, '*/')), key=os.path.getmtime))
        print("*******")
        self.outputs["timestamp_folder"].write(max(glob.glob(os.path.join(DS_DATA_HOME, '*/')), key=os.path.getmtime))
        print("********")
        pool.close()
        pool.join()
           
class KeywordExtraction(BaseOperator):

    @property
    def inputs(self):
        return {"content_meta": Pandas_Dataframe(self.node.inputs[0]),
                "taxonomy": Pandas_Dataframe(self.node.inputs[1]),           
                "timestamp_folder": File_Txt(self.node.inputs[2])

                }

    @property  # how to write to a folder?
    def outputs(self):
        return {"path_to_contentKeywords": File_Txt(self.node.outputs[0]) 
                }

    def keyword_extraction_parallel(self, dir, path_to_texts, taxonomy, extract_keywords, method, filter_criteria):
        print("*******dir*********:", dir)
        print("***Extract keywords***:", extract_keywords)
        print("***Filter criteria:***", filter_criteria)
        path_to_cid_transcript = os.path.join(path_to_texts, dir, "enriched_text.txt")
        keywords = os.path.join(path_to_texts, dir, "keywords")
        path_to_text_tokens = os.path.join(keywords, "text_tokens")
        path_to_tagme_tokens = os.path.join(keywords, "tagme_tokens")
        path_to_tagme_taxonomy_intersection = os.path.join(keywords, "tagme_taxonomy_tokens")

        if os.path.isfile(path_to_cid_transcript):
            logging.info("Transcript present for cid: {0}".format(dir))
            try:

                # text_file = os.listdir(path_to_cid_transcript)[0]
                if os.path.getsize(path_to_cid_transcript) > 0:
                    print("Path to transcripts ", path_to_cid_transcript)
                    logging.info("Running keyword extraction for {0}".format(path_to_cid_transcript))
                    logging.info("---------------------------------------------")

                    if extract_keywords == True and filter_criteria == "none":
                        logging.info("Tagme keyword extraction is running for {0}".format(path_to_cid_transcript))
                        path_to_pafy_tagme_tokens = get_tagme_longtext(path_to_cid_transcript, path_to_tagme_tokens)
                        logging.info("Path to tagme tokens is {0}".format(path_to_pafy_tagme_tokens))

                    elif extract_keywords == False:
                        logging.info("Text tokens extraction running for {0}".format(path_to_cid_transcript))
                        path_to_pafy_text_tokens = pafy_text_tokens(path_to_cid_transcript, path_to_text_tokens)
                        logging.info("Path to text tokens is {0}".format(path_to_pafy_text_tokens))

                    elif extract_keywords == True and filter_criteria == "taxonomy_keywords":
                        logging.info("Tagme intersection taxonomy keyword extraction is running for {0}".format(
                            path_to_cid_transcript))
                        revised_content_df = pd.read_csv(taxonomy, sep=",", index_col=None)
                        clean_keywords = map(get_words, list(revised_content_df["Keywords"]))
                        clean_keywords = map(clean_string_list, clean_keywords)
                        flat_list = [item for sublist in list(clean_keywords) for item in
                                     sublist]
                        taxonomy_keywords_set = set([cleantext(i) for i in flat_list])
                        path_to_pafy_tagme_tokens = get_tagme_longtext(path_to_cid_transcript, path_to_tagme_tokens)
                        path_to_tagme_intersect_tax = tagme_taxonomy_intersection_keywords(taxonomy_keywords_set,
                                                                                           path_to_pafy_tagme_tokens,
                                                                                           path_to_tagme_taxonomy_intersection)
                        logging.info \
                            ("Path to tagme taxonomy intersection tokens is {0}".format(path_to_tagme_intersect_tax))

                    else:
                        logging.info("Invalid argument provided")


                else:
                    logging.info("The text file {0} has no contents".format(path_to_cid_transcript))
                    print("The text file {0} has no contents".format(path_to_cid_transcript))

            except:
                print("Raise exception for {0} ".format(path_to_cid_transcript))
                logging.info("Raise exception for {0} ".format(path_to_cid_transcript))
        else:
            logging.info("Transcripts doesnt exist for {0}".format(path_to_cid_transcript))
            print("Transcripts doesnt exist for {0}".format(path_to_cid_transcript))

        return path_to_texts


    def run(self, extract_keywords, method, filter_criteria):
        assert method=="tagme"

        content_meta = self.inputs["content_meta"].read()
        taxonomy = self.inputs["taxonomy"].read()
        timestamp_folder = self.inputs["timestamp_folder"].read()
        print("****timestamp folder:", timestamp_folder)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        content_to_text = timestamp_folder+"content_to_text"
        print("content_to_text path:", content_to_text)
        if not os.path.exists(content_to_text):
            logging.info("No such directory as: ", content_to_text)
        
        else:  
            logging.info('------Transcripts to keywords extraction-----')
            logging.info("Keyword extraction method used:-{0}".format(method))
            
            pool = multiprocessing.Pool(processes=4)
            keywordExtraction_partial = partial(self.keyword_extraction_parallel,
                                                path_to_texts=content_to_text, taxonomy=taxonomy, extract_keywords=extract_keywords, method=method, filter_criteria=filter_criteria)  # prod_x has only one argument x (y is fixed to 10)
            results = pool.map(keywordExtraction_partial, [dir for dir in os.listdir(content_to_text)])
            print("path to content keywords:", max(glob.glob(os.path.join(timestamp_folder[:-1], 'content_to_text'))))
            self.outputs["path_to_contentKeywords"].write(max(glob.glob(os.path.join(timestamp_folder[:-1], 'content_to_text')), key=os.path.getmtime))

            pool.close()
            pool.join()

class CorpusCreation(BaseOperator):

    @property
    def inputs(self):
        return { "taxonomy": Pandas_Dataframe(self.node.inputs[0]),
                 "path_to_contentKeywords": File_Txt(self.node.inputs[1])          
                }

    @property  # how to write to a folder?
    def outputs(self):
        return {"root_path": File_Txt(self.node.outputs[0]),
                "path_to_corpus": File_Txt(self.node.outputs[1])
                }

    def run(self, keyword_subfolder, update_corpus, delimiter):
        assert keyword_subfolder == "tagme_tokens" or keyword_subfolder == "text_tokens" or keyword_subfolder == "tagme_taxonomy_tokens"

        taxonomy = self.inputs["taxonomy"].read()
        path_to_contentKeywords = self.inputs["path_to_contentKeywords"].read()
        corpus_folder = os.path.split(path_to_contentKeywords)[0]+"/corpus"
        if not os.path.exists(corpus_folder):
            os.makedirs(corpus_folder)
        root_path = os.path.split(os.path.split(path_to_contentKeywords)[0])[0]
        corpus_loc=os.path.join(root_path,"corpus.csv")
        #keyword_dir=timestamp_folder+"content_to_text"
        vocabulary_loc=os.path.join(corpus_folder,"vocab")
        cids=os.listdir(path_to_contentKeywords)
        
        content_keywords_list = []

        for content in cids:
            
            path_to_keywords=os.path.join(path_to_contentKeywords,content,"keywords", keyword_subfolder, "keywords.csv")##

            if not os.path.exists(path_to_keywords):
                extracted_keys = []
            else:
                extracted_keyword_df=pd.read_csv(path_to_keywords, keep_default_na=False)
                extracted_keys=list(extracted_keyword_df['KEYWORDS'])
            
            content_keywords_list.append(extracted_keys)
            print("content_keywords_list: ",content_keywords_list)

        content_keywords_list=custom_listPreProc(content_keywords_list,'stem_lem', delimiter)

        taxonomy['Keywords']=[get_words(i) for i in list(taxonomy['Keywords'])]  #get words from string of words
        taxonomy_keywords=[x for x in list(taxonomy['Keywords']) if str(x) != 'nan']  # remove nan
        taxonomy_keywords=custom_listPreProc(taxonomy_keywords,'stem_lem', "_")

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

class ContentTaxonomyMapping(BaseOperator):

    @property
    def inputs(self):
        return {"content_meta": Pandas_Dataframe(self.node.inputs[0]),
                "taxonomy": Pandas_Dataframe(self.node.inputs[1]),
                "root_path": File_Txt(self.node.inputs[2]),
                "path_to_corpus": File_Txt(self.node.inputs[3])

                }

    @property  # how to write to a folder?
    def outputs(self):
        return {"path_to_timestampFolder": File_Txt(self.node.outputs[0]),
                "path_to_distMeasure": File_Txt(self.node.outputs[1])
                }

    def run(self, grade_start, grade_end, keyword_subfolder, level, delimiter, phrase_split, distanceMeasure, embedding_method):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        assert keyword_subfolder == "tagme_tokens" or keyword_subfolder == "text_tokens" or keyword_subfolder == "tagme_taxonomy_tokens"

        print("****grade_start:", grade_start)
        print("****grade_end:",  grade_end)
        print("****keyword_subfolder:", keyword_subfolder)
        print("****level:", level)
        print("****delimiter:", delimiter)
        print("****phrase_split:", phrase_split)
        print("****distanceMeasure:", distanceMeasure)
        print()
        
        content_meta = self.inputs["content_meta"].read()
        taxonomy = self.inputs["taxonomy"].read()
        root_path = self.inputs["root_path"].read()
        corpus_folder = self.inputs["path_to_corpus"].read()
        #check for the presence of corpus folder:
        if not os.path.exists(corpus_folder):
           logging.info("No corpus folder created")
        else:
            vocab_loc = corpus_folder+"/vocab"
            vocabulary=load_obj(vocab_loc)
        

        mapping_folder = root_path+"/content_taxonomy_mapping"

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


        DELIMITTER = delimiter
        # cleaning taxonomy KEYWORDS
        taxonomy['Keywords']=[get_words(item) for item in list(taxonomy['Keywords'])]  #get words from string of words
        taxonomy_keywords=[x for x in list(taxonomy['Keywords']) if str(x) != 'nan']  # remove nan
        taxonomy_keywords=custom_listPreProc(taxonomy_keywords,'stem_lem',DELIMITTER) #??

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
                if not os.path.exists(os.path.join(path_to_corpus ,content, "keywords", keyword_subfolder, "keywords.csv")):
                    extracted_keys = []
                else:
                    extracted_keyword_df =pd.read_csv(os.path.join(path_to_corpus ,content, "keywords", keyword_subfolder, "keywords.csv"), keep_default_na=False)
                    print("keywords:", list(extracted_keyword_df['KEYWORDS']))
                    extracted_keys =list(extracted_keyword_df['KEYWORDS'])
                content_keywords_list.append(extracted_keys)
            print("*****keyword list:", content_keywords_list)

            content_keywords_list=custom_listPreProc(content_keywords_list,'stem_lem', DELIMITTER) #??
            content_meta['Content_keywords'] =content_keywords_list
            content_meta = content_meta.iloc[[i for i ,e in enumerate(content_meta['Content_keywords']) if (e!=[] and len(e)>5)]]
            content_meta = content_meta.reset_index(drop=True)

            # specify the level at which the aggregation needs to take place
            # level = level
            domains = list(set(content_meta["subject"]) & set(taxonomy["Subject"]))
            print("content meta columns: ", content_meta.columns)
            print("taxonomy columns:", taxonomy.columns)
            print("Domains: ", domains)

            logging.info("Aggregated on level: {0}".format(level))
            logging.info("------------------------------------------")

            dist_all =dict()
            for i in domains:
                subject = [i]

                logging.info("Running for subject: {0}".format(subject))
                domain_content_df =content_meta.loc[content_meta['subject'].isin(subject)]
                domain_content_df.index =domain_content_df['identifier']
                domain_taxonomy_df =taxonomy.loc[taxonomy['Subject'].isin(subject)]
                level_domain_taxonomy_df =get_level_keywords(domain_taxonomy_df ,level)
                if (distanceMeasure=='jaccard1' or distanceMeasure=='jaccard2'):
                    print("****", level_domain_taxonomy_df) 
                    level_domain_taxonomy_df.index =level_domain_taxonomy_df[level]

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
                            dist_all['& '.join(subject)] = dist_df    
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
                        taxonomy_freq_df=pd.DataFrame(taxonomy_freq_df.todense(), index=list(level_domain_taxonomy_df[level]), columns=vectorizer.get_feature_names())

                        content_freq_df=vectorizer.transform(content_documents)
                        content_freq_df=pd.DataFrame(content_freq_df.todense(), index=list(domain_content_df.index), columns=vectorizer.get_feature_names())
                        dist_df=pd.DataFrame(cosine_similarity(content_freq_df,taxonomy_freq_df), index=list(domain_content_df.index), columns=list(level_domain_taxonomy_df[level]))
                        dist_all['& '.join(subject)] = dist_df 


            if not os.path.exists(output):
                os.makedirs(output)
            save_obj(dist_all, os.path.join(output,distanceMeasure+"_dist_all"))
            self.outputs["path_to_timestampFolder"].write(root_path)
            self.outputs["path_to_distMeasure"].write(os.path.join(output,distanceMeasure+"_dist_all"))


class EvlnKnownTags(BaseOperator):

    @property
    def inputs(self):
        return {"content_meta": Pandas_Dataframe(self.node.inputs[0]),
                "taxonomy": Pandas_Dataframe(self.node.inputs[1]),
                "path_to_timestampFolder": File_Txt(self.node.inputs[2])
                }

    @property  # how to write to a folder?
    def outputs(self):
        return {"path_to_EvaluationMatrix": File_Txt(self.node.outputs[0])
                }

    def run(self, window, distanceMeasure, level, tax_known_tag, content_known_tag):

        if distanceMeasure=="jaccard1" or distanceMeasure=="jaccard2":
            sort_order=0
        if distanceMeasure=="cosine":
            sort_order=1
        level_pred_dict=dict()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        content_meta = self.inputs["content_meta"].read()
        taxonomy = self.inputs["taxonomy"].read()
        timestamp_folder = self.inputs["path_to_timestampFolder"].read()
        
        output=timestamp_folder+"/content_taxonomy_mapping"
        if not os.path.exists(output):
            logging.info("Taxonomy mapping not performed")
        else:
            evaluation_output = os.path.join(os.path.split(timestamp_folder)[0]+"/evaluation",timestr)


            if not os.path.exists(evaluation_output):
                os.makedirs(evaluation_output)
                
            dist_dict_list=[load_obj(os.path.join(output, path_to_runFolder, distanceMeasure +"_dist_all")) for path_to_runFolder in os.listdir(output) if os.path.exists(os.path.join(output, path_to_runFolder, distanceMeasure+"_dist_all.pkl"))]

            dist_dict = dictionary_merge(dist_dict_list)
            print(dist_dict)

            eval_dct=dict()
            for subject in dist_dict.keys():
                pred_df=dist_dict[subject]
                known_tag_ind=[]
                for item in list(list(list(pred_df.columns))):
                    known_tag_ind.append(getGradedigits(taxonomy[tax_known_tag][list(taxonomy[level]).index(item)]))
                pred_df.columns=known_tag_ind
                actual_df=pd.DataFrame({content_known_tag:[getGradedigits(content_meta[content_known_tag][list(content_meta['identifier']).index(i)]) for i in list(pred_df.index) ]}, index=pred_df.index)
                eval_dct[subject]=list(getEvalMatrix(pred_df, actual_df, content_known_tag,sort_order,window)['percent'])

            pd.DataFrame(eval_dct).to_csv(os.path.join(evaluation_output,distanceMeasure+"_evalMatrix.csv"))
            with open(os.path.join(evaluation_output, "evaluation_info.txt"), "w") as eval_info:
                eval_info.writelines("DistanceMeasure: {0} \n".format(distanceMeasure))
                eval_info.writelines("Evaluation criteria: {0} \n".format("Evaluation based on known tags"))
                eval_info.writelines("Window: {0} \n".format(window))
                eval_info.writelines("Evaluation matrix saved in: {0} \n".format(os.path.join(evaluation_output,distanceMeasure+"_evalMatrix.csv")))
            self.outputs["path_to_EvaluationMatrix"].write(os.path.join(evaluation_output,distanceMeasure+"_evalMatrix.csv"))


                    