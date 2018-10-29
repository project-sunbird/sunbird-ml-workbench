import difflib 
import daggit 
import yaml 
from daggit.contrib.ekstep.operators.contentTaggingUtils import *



def read_yaml(data_location):
  with open(data_location, 'r') as stream:
    data = yaml.load(stream)
  return data  

def sentence_similarity(sentence1, sentence2, threshold):       # sentence simil
  sentence = difflib.SequenceMatcher(lambda x: x == " ",
                     sentence1, 
                     sentence2) 
  similarity_score = sentence.ratio()*100 
  if similarity_score >= threshold: 
    return 1
  else:
    return 0 


def intersection_lists(list_1, list_2, threshold):
    if 1.0*(len(set(list_1) & set(list_2))/ min(len(set(list_1)),len(set(list_2)))) > threshold:
        return 1
    else:
        return 0 

def keyword_extraction( path_to_text, path_to_save_tagme, expected_output):
    file_ = open(path_to_text, "r")
    text = file_.readline()
    if text == '':
        return "Text is not available"
    else:
      path_to_tagme_output = get_tagme_longtext(path_to_text, path_to_save_tagme)    
      actual_output = pd.read_csv(path_to_tagme_output)
      actual_output = list(actual_output["KEYWORDS"])
      intersection_lists_output = intersection_lists(actual_output, expected_output,0.8)  
      return intersection_lists_output


def jaccard_evaluation(content_keywords, taxonomy_keywords, evaluation_criteria, threshold):
    evaluation_score = jaccard_with_phrase(content_keywords, taxonomy_keywords)
    if evaluation_score[evaluation_criteria] < threshold:
        return 1          
    else:
        return 0 

