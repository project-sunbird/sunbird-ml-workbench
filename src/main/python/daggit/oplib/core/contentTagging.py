import multiprocessing
import time
import os
import logging

from daggit.iolib.io import Pandas_Dataframe, Read_Folder
from daggit.core.base import BaseOperator
from ..operators_registry import get_op_callable
from ..core.contentTaggingUtils import clean_url, identify_fileType, content_to_text_conversion, contentTotext_EndtoEnd
from ..core.contentTaggingUtils import keyword_extraction_parallel
class ContentToText(BaseOperator):

    @property
    def inputs(self):
        return {"content_meta": Pandas_Dataframe(self.node.inputs[0])
                }

    @property #how to write to a folder?
    def outputs(self):
        return {"path_to_corpus": Read_Folder(self.node.outputs[0]),
                "path_to_LogFile": Read_Folder(self.node.outputs[1])
                }

    def run(self, range_start, range_end, parallelise, url_type, youtube_extraction, ecml, pdf):
        content_meta = self.inputs["content_meta"].read()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        content_to_text = self.outputs["path_to_corpus"].read_loc()
        log_live = self.outputs["path_to_LogFile"].read_loc()
        
        #content dump:
        if not os.path.exists(content_to_text):
            os.makedirs(content_to_text)
            print("content_to_text: ", content_to_text)

        # log_file:
        log_file = os.path.join(log_live, "content_to_text", timestr)
        if not os.path.exists(log_file):
            os.makedirs(log_file)
        
        # setting up logging
        logging.basicConfig(filename=os.path.join(log_file, "content_to_text.log"), filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s')
    
        logging.info("CTT_CONTENT_TO_TEXT_START")
        logging.info(
            "CTT_Config: content_meta from range_start: {0} to range_end: {1} created in: {2}".format(
                range_start, range_end, content_to_text))
        
        #read content meta:
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
        log_live = os.path.join(log_file, "Content_to_text.log")
        logging.basicConfig(filename=log_live, filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')
       
        logging.info('Number of Content detected in the content meta: ' + str(len(content_meta)))
        logging.info(
            "-----Running Content to Text for contents from index {0} to index {1}:-----".format(range_start, range_end))
        logging.info("time started: {0}".format(start))
        
        #check for youtube url type:-
        if url_type == "youtube":
            downloadField = youtube_extraction['contentDownloadField']

        elif url_type == "ecml":
            downloadField = ecml['contentDownloadField']

        else:
            downloadField = pdf['contentDownloadField']
        
        content_meta = content_meta[content_meta["content_type"] == url_type]
        content_meta.reset_index(drop=True, inplace=True)

        print("DownloadField: ", downloadField)
        print("Need to parallelise or not: ", parallelise)
        
    
        
        if parallelise == True:
            print("Parallelising!!!!!")
            if __name__ == '__main__':
                indices = [i for i in range(range_start, range_end)]
                pool = multiprocessing.Pool(processes=4)
                contentTotext_partial = partial(contentTotext_EndtoEnd, content_meta=content_meta,
                                                downloadField=downloadField, content_to_text=content_to_text)  # prod_x has only one argument x (y is fixed to 10)
                results = pool.map(contentTotext_partial, indices)
        
                pool.close()
                pool.join()
        else:
            for i in range(range_start, range_end):
                contentTotext_EndtoEnd(i, content_meta, downloadField, content_to_text)

class KeywordExtraction(BaseOperator):

    @property
    def inputs(self):
        return {"content_meta": Pandas_Dataframe(self.node.inputs[0]),
                "path_to_corpus": Read_Folder(self.node.outputs[1]),
                "taxonomy": Pandas_Dataframe(self.node.inputs[2])
                }

    @property #how to write to a folder?
    def outputs(self):
        return {"path_to_corpus": Read_Folder(self.node.outputs[0]),
                "path_to_LogFile": Read_Folder(self.node.outputs[1])
                }

    def run(self, extract_keywords, method, filter_criteria):
        content_meta = self.inputs["content_meta"].read()
        taxonomy = self.inputs["taxonomy"].read()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        content_to_text = self.outputs["path_to_corpus"].read_loc()
        log_live = self.outputs["path_to_LogFile"].read_loc()
        
        log_file = os.path.join(log_live, "keyword_extraction", timestr)
        if not os.path.exists(log_file):
            os.makedirs(log_file)
       
        logging.basicConfig(filename=os.path.join(log_file, "keyword_extraction.log"), filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s')
        logging.info('------Transcripts to keywords extraction-----')
        logging.info("Keyword extraction method used:-{0}".format(method))

        if __name__ == "__main__":
            pool = multiprocessing.Pool(processes=4)
            keywordExtraction_partial = partial(keyword_extraction_parallel,
                                                path_to_text=content_to_text,taxonomy=taxonomy)  # prod_x has only one argument x (y is fixed to 10)
          
            results = pool.map(keywordExtraction_partial, [dir for dir in os.listdir(content_to_text)])
            pool.close()
            pool.join()

                          



