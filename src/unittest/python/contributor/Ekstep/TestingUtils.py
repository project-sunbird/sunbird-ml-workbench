import difflib 
import daggit 
import yaml 
from daggit.contrib.ekstep.operators.contentTaggingUtils import *



def read_yaml(data_location):
  with open(data_location, 'r') as stream:
    data = yaml.load(stream)
  return data  

def content_meta_features_checking(data_location, mandatoy_fields_location):
  read_data = read_yaml(data_location)
  read_mandatoy_fields = read_yaml(mandatoy_fields_location)
  mandatoy_fields = list(read_mandatoy_fields['mandatory_fields']) 
  check = [0 if elem in list(read_data.keys()) else 1 for elem in mandatoy_fields]
  if sum(check) > 0:
    return 0
  else:
    return 1

def sentence_similarity(sentence1, sentence2, threshold):       # sentence simil
  sentence = difflib.SequenceMatcher(lambda x: x == " ",
                     sentence1, 
                     sentence2) 
  similarity_score = sentence.ratio()*100 
  if similarity_score >= threshold: 
    return 1
  else:
    return 0 

def text_Extraction(url, type_of_url, id_name, path_to_save, expected_text):  
    text_generated_path = content_to_text_conversion(url, type_of_url, id_name, path_to_save)
    actual_text = open(text_generated_path, "r") 
    actual_text = actual_text.read()
    if sentence_similarity(actual_text, expected_text, 0.90) == 1:
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
    
    if detect(text) != 'en':           
        return "Non English text"
    else: 
        path_to_tagme_output = get_tagme_longtext(path_to_text, path_to_save_tagme)    
        actual_output = pd.read_csv(path_to_tagme_output)
        actual_output = list(actual_output["KEYWORDS"])
        intersection_lists_output = intersection_lists(actual_output, expected_output,0.8)  
        return intersection_lists_output


def jaccard_evaluation(content_keywords, taxonomy_keywords, evaluation_criteria, threshold):
    evaluation_score = jaccard_with_phrase(content_keywords, taxonomy_keywords)
    if evaluation_score[evaluation_criteria] < threshold:
        return 1          #### Jaacard value, 
    else:
        return 0 

