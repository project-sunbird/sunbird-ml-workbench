import io
import requests
import os
import logging

#change if it doesnt work
logging.getLogger("requests").setLevel(logging.WARNING) 

import signal
import re
import sys
import json
import zipfile
import unicodedata
import time
import csv
import string
import shutil
import urllib
from urllib.parse import urlparse
import math
# import StringIO
import pickle
import youtube_dl

import mutagen.mp3 as mp3
import speech_recognition as sr
import pandas as pd
import numpy as np
import googleapiclient.discovery
import nltk
nltk.download("wordnet")
nltk.download("stopwords")
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import time


from google.cloud import translate
from google.cloud import vision
from pydub.playback import play
from PyPDF2 import PdfFileReader
import Levenshtein
import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go

from pydub import AudioSegment
from natsort import natsorted
from plotly import tools
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer



GOOGLE_APPLICATION_CREDENTIALS = r"""{
  "type": "service_account",
  "project_id": "machinelearningapi-175409",
  "private_key_id": "27cc5d0e439a6eeef7dd2a7c35c00d3dc4acd953",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCSFKC2+dMG6qhz\nx0smAFQGMXCWtiuti75G0Gurn3ciG/wEKsUWgd/dquFBJYB4ZOjOsyq0S03NRzFH\n/zwENFUlpitBMNbLoVCuL3x8rU5BKTN8xAu0RRsj/kmV/+zgiChVPgpstVQppNtf\nuc5+cPYJmUwrQ6t96zqhjppt0VhDiOtfIfCfXi5LK6Wj1T3i2AQL9Ekuaw1o2QDF\ncQ+EGb7yUkCy1MYGZ7KWtx7MhYIARL7/91Y1ry1mE00n48pmmMe6KPuC/Z0RY/X3\nrxFl6YScMm0AlP/r6gYmOHFI/d/SkJtDB4LWAmIiG80iK5X2pnErMt5ezyGqhFIk\nGhUizhFfAgMBAAECggEAAQbuXMeA4zNNFMVmNGJ0uljapCVTOilnimKHXeEbq1Lb\nC+MQDWvZvDkqC2ixZWoWef5DzJmWa5S387kz8SugJ7hkrUnK4Swpnlxs2aSaeCmZ\nNtB8fUauEQp1RJmthq9Px+THsD6We5hMs5oewBOeM8EpeZbVKyU3S1wE4cTIV5CC\ngmcwz3y+yttplREeAp7So9V0R5FpJ7P0qjal3Ngyu5aR6pgObNzhjXKeed6aULwc\neZIDlCwJhxetLqyhsGHHvFPQas6Y5mnET5mA4KRLV3u9KBh5KFc2f93nk7nqPeWz\nf0AQH9avBvTFT/RLBGRH4jNGaMovce0+AI9QLORzcQKBgQDEICcztYPIVjWvZ0Rs\ncBmhuy7yD1Rh7pZisC4A5iwoG7JH8sBXJm33bMWmRlztVDSLxi0GTS4ha7cGTSxG\nVJBKCUwVMWKDQQXQs9SKlUnTF94ozRzyozq9MSrJJcsXs9dE0++cTpT54eD6n8dU\nvLmPkrJ4j4u7SftgJQiQi9oUvQKBgQC+rUuQeinPpAXOKXsPPSOF/Z+kn8SS/kG8\nnkRqcsCnPOZxXUArBMmTahn/bUI6T+PzlXRbQSzEF3N1U++YquTqxACrwkiBv9Ak\ny7XugZzBcCmv4bDYgspOo0emKyl5uAvmQjceYgVANTsC5pD7hQERNGKvl8MA9VZn\nNlSA2iTWSwKBgQC/YRy/4Z0RzcYfPibPpeftIOnjfL/7vER1UrPhXrmx/azPdnrn\nz/E4oqSP51Ngp22LAzwGTSP5qtFzTbUpf/U4ua/LcmBN8hJJoGGDRcA/Q6geqmBY\nCJ4V5bd5hu6SV4R1flXvceL/n8HY7jclYe+0wRJ0gKZ6gOvR2vFrk3ygBQKBgBeg\nr8Fqce3p/FIsr7QWtmUvJW4n4hr46LpvvjiWmarfkAqyLHZoNHZQ6oHNTyyco7mW\nZoG8VMjDwynhycnYO1+gBBlEjOmPFELK/3NbmkoaFQBXbiuWIW2XLBS6Onx7wvW4\ndM4OBWqMbhCQ85xHQfeYzzXFD4P54sgNYnFJFtF7AoGAbMlWoReAAnjfB78chUZU\n8HWcSXi8ZLhR/uaZY7S3yJwA2Kz7Wa05WJ6hhT6MWHojClH36ZZb8jms/ddFKEuG\nTEaLGEL3eoniYtzg7wk34u9H6+0OBhXdYJbXcnapeUR4FYhOEw/9AA3dNhD2/ne+\npVGBX9PagFWhBI1TC75klnM=\n-----END PRIVATE KEY-----\n",
  "client_email": "machinelearningapi@machinelearningapi-175409.iam.gserviceaccount.com",
  "client_id": "107107468801573946683",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://accounts.google.com/o/oauth2/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/machinelearningapi%40machinelearningapi-175409.iam.gserviceaccount.com"
}"""


pun_list = list(string.punctuation)


def downloadZipFile(url,directory):
    
    r=requests.get(url)
    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(directory)
        return r.ok
    except:
        return False


#The shutil module includes high-level file operations such as copying, setting permissions,

def findFiles(directory,substrings):
    ls=[]
    if type(directory) == str and type(substrings) == list:
    # assert type(directory)==unicode or type(directory)==str
    # assert type(substrings)==list
        if os.path.isdir(directory):
            for dirname, dirnames, filenames in os.walk(directory):
                print(dirname, dirnames, filenames)
                for filename in filenames:
                    string=os.path.join(dirname, filename)
                    for substring in substrings:
                        if(string.find(substring)>=0):
                            ls.append(string)
    return ls


def unzip_files(directory):
    assert type(directory)==str
    #Finds all files in a directory that are of type .zip
    zip_list=findFiles(directory,['.zip'])
    bugs={}
    for zip_file in zip_list:
      #In case zipfile is bad
      try:
        #Extract zip file
        with zipfile.ZipFile(zip_file, 'r') as z:
          z.extractall(directory)
        #Delete zip file after extraction
        os.remove(zip_file)
      except:
        #Can return bugs if you want list of buggy zip files
        bugs.append(zip_file)
        
        
#Transfer the files in assets,data,items and the ecml files
def copy_main_folders(root,identifier):
    assert type(identifier)==str
    assert type(root)==str
    #List of files to be copied (To flatten directory structure)
    file_list=findFiles(os.path.join(root,'temp'+identifier),['asset','data','item','ecml'])
    path=os.path.join(root,identifier)
    #To make the new directory in which files will be eventually stored
    if not os.path.exists(path):
        os.makedirs(path)

    #To make the new sub-directories in which the files will be eventually stores
    location=[os.path.join(path,folder) for folder in ['assets','data','items']]
    for loc in location:
        if not os.path.exists(loc):
            os.makedirs(loc)
    #Copying files
    for f in file_list:
      if(f.find('asset')>=0):
          shutil.copy(f,os.path.join(path,'assets'))
      elif(f.find('data')>=0):
          shutil.copy(f,os.path.join(path,'data'))
      elif(f.find('item')>=0):
          shutil.copy(f,os.path.join(path,'items'))
      else:
          shutil.copy(f,path)
    #Delete the messy download directory
    shutil.rmtree(os.path.join(root,'temp'+identifier))


def copy_main_folders_new(root, identifier):
    assert type(identifier)==str
    assert type(root)==str
    download_dir = os.path.join(root,'temp'+identifier)
    path=os.path.join(root,identifier)
    
    #To make the new directory in which files will be eventually stored
    if not os.path.exists(path):
        print("path:", path)
        os.makedirs(path)

    #To make the new sub-directories in which the files will be eventually stores
    location=[os.path.join(path,folder) for folder in ['assets','data','items']]

    for loc in location:
        if not os.path.exists(loc):
            os.makedirs(loc)
    for d in os.listdir(os.path.join(root,'temp'+identifier)):
        
        if os.path.isdir(os.path.join(download_dir, d)) and len(os.listdir(os.path.join(download_dir, d)))>0:
            
            for i in os.listdir(os.path.join(download_dir, d)):
                shutil.copy(os.path.join(download_dir, d, i) , os.path.join(path, "assets"))
        else:
            
            shutil.copy(os.path.join(download_dir, d) , path)
    #Delete the messy download directory
    shutil.rmtree(os.path.join(root,'temp'+identifier))


def getImgTags(img_file_name):
    # Instantiates a client
    vision_client = vision.ImageAnnotatorClient()
    with io.open(img_file_name, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)
    response = vision_client.text_detection(image)
    texts = response.text_annotations
    full_text = []
    for text in texts:
        full_text.append(text.description)
    img_dct={'text': full_text[0]}   
    return img_dct


def clean_url(my_url):
    """ Clean the url to be made into 
    content_id to use it as a mapping feature for our content """   
    intab = "!@#$%^&*()[]{};:,./<>?\|`~-=_+"
    for ch in intab:
        my_url = my_url.replace(ch,"_")
    return my_url


def cleantext(text):
     replace_char=["[","]","u'","None","Thank you","-","(",")","#","Done" , ">" ,"<","-" ,"|","/","\"","Hint","\n","'"]

     for l in replace_char:
         text = text.replace(l,"")
         text = re.sub(' +',' ',text)
     return text


def word_proc(word):
    delimitters=[" ",",","_","-",".","/"] + list(set(string.punctuation))
    #print word
    for lim in delimitters:
       
        word=word.replace(lim,"_")
        #word=word.encode('ascii', 'ignore').decode('ascii')
    return word


def identify_fileType(url):
  
    if url.startswith('https://www.youtube.com') or url.startswith('https://youtu.be'):
        if "embed" in url:
            return "youtube_embed"
        else:
            return "youtube"
    elif url.endswith('pdf'):
        return "pdf"
    elif  url.endswith('ecar') or url.endswith("zip"):
        return "ekstep_archived"
    elif url.startswith('https://drive.google.com') or url.startswith('https://en.wikipedia.org'):
        return "html"
    else:
        return "unknown"


def fetch_video_id(url):
    parse_url = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parse_url.query)
    return query["v"][0]


def embed_youtube_url_validation(url):
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

    youtube_regex_match = re.match(youtube_regex, url)
    
    if youtube_regex_match:
        if youtube_regex_match.group(6) == "https://www":

            return "https://www.youtube.com/watch?v="+fetch_video_id(url.split("embed/")[1])[:11]

        else:
            return "https://www.youtube.com/watch?v="+youtube_regex_match.group(6)

def multimodel_text_enrichment(index, content_meta, downloadField, content_to_text_path):
    url = content_meta[downloadField][index]
    # content_to_text = os.path.join(DS_DATA_HOME, "output", "content_to_text")
    # pathTotext_corpus = os.path.join(DS_DATA_HOME, timestr, "textCorpus")
    content_id = content_meta["identifier"][index]
    logging.info("CTT_CONTENT_ID_READ: {0}".format(content_id))

    fileType = identify_fileType(url)

    print("content_to_text path is: ", content_to_text_path)
    print("fileType of url [{0}] is: {1}". format(url, fileType))

    logging.info("CTT_CONTENT_TYPE: {0}".format(fileType))
    logging.info("-----Text Extraction for url: {0}".format(url))
    logging.info("CTT_TEXT_EXTRACTION_START")
    path_to_text = content_to_text_conversion(url, fileType, content_id, content_to_text_path)
    return path_to_text
    logging.info("CTT_CONTENT_TO_TEXT_STOP")

def keyword_extraction_parallel(dir, path_to_text, taxonomy):
    
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

                if extract_keyword == 1 and filter_criteria == "none":
                    logging.info("Tagme keyword extraction is running for {0}".format(path_to_cid_transcript))
                    path_to_pafy_tagme_tokens = get_tagme_longtext(path_to_cid_transcript, path_to_tagme_tokens)
                    logging.info("Path to tagme tokens is {0}".format(path_to_pafy_tagme_tokens))

                elif extract_keyword == 0:
                    logging.info("Text tokens extraction running for {0}".format(path_to_cid_transcript))
                    path_to_pafy_text_tokens = pafy_text_tokens(path_to_cid_transcript, path_to_text_tokens)
                    logging.info("Path to text tokens is {0}".format(path_to_pafy_text_tokens))

                elif extract_keyword == 1 and filter_criteria == "taxonomy_keywords":
                    logging.info("Tagme intersection taxonomy keyword extraction is running for {0}".format(
                        path_to_cid_transcript))
                    revised_content_df = pd.read_csv(taxonomy, sep=",", index_col=None)
                    clean_keywords = map(get_words, list(revised_content_df["Keywords"]))
                    clean_keywords = map(clean_string_list, clean_keywords)
                    flat_list = [item.decode("utf-8").encode("ascii", "ignore") for sublist in list(clean_keywords) for item in
                                 sublist]
                    taxonomy_keywords_set = set([cleantext(i) for i in flat_list])
                    path_to_pafy_tagme_tokens = get_tagme_longtext(path_to_cid_transcript, path_to_tagme_tokens)
                    path_to_tagme_intersect_tax = tagme_taxonomy_intersection_keywords(taxonomy_keywords_set,
                                                                                       path_to_pafy_tagme_tokens,
                                                                                       path_to_tagme_taxonomy_intersection)
                    logging.info("Path to tagme taxonomy intersection tokens is {0}".format(path_to_tagme_intersect_tax))

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


def url_to_audio_extraction(url, path, type):
    logging.info("UTAE_YOUTUBE_URL_START: {0}".format(url))
    print("current_directory_UTAE", os.getcwd())
    if not os.path.exists(path):
        os.makedirs(path)
    
    cid  = os.path.split(os.path.split(path)[0])[1]
    path_to_audio = os.path.join(path, cid +".mp3")
    print(path_to_audio)
    logging.info("UTAE_YOUTUBE_URL_START: {0}".format(url))
    if not os.path.isfile(path_to_audio):

        if type == "youtube":
            os.chdir(path)
            ydl_opts = {
                            'format': 'bestaudio[asr=44100]/best',
                            'postprocessors': [{
                                'key': 'FFmpegExtractAudio',
                                'preferredcodec': 'mp3',
                                'preferredquality': '256'
                            }]
                        }

            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            os.rename(list(filter((lambda x: '.mp3' in x), os.listdir(path)))[0], path_to_audio)
            logging.info("UTAE_AUDIO_DOWNLOAD_COMPLETE")

        if type == "youtube_embed":
           url = embed_youtube_url_validation(url)
           os.chdir(path)
           ydl_opts = {
                            'format': 'bestaudio[asr=44100]/best',
                            'postprocessors': [{
                                'key': 'FFmpegExtractAudio',
                                'preferredcodec': 'mp3',
                                'preferredquality': '256'
                            }]
                        }

           with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
           os.rename(list(filter((lambda x: '.mp3' in x), os.listdir(path)))[0], path_to_audio)
           return path_to_audio

           logging.info("UTAE_AUDIO_DOWNLOAD_COMPLETE")

        else:
            logging.info("Type unknown")
            print("Type is Unknown")
        
        return path_to_audio

      
    else:
        #return os.path.join(path_to_audio)
        return path_to_audio
    logging.info("UTAE_YOUTUBE_URL_STOP")


def audio_split(path_to_audiofile, path_to_split_audio):
    """Takes in an audiofile, split it as per the duration and returns the path to the split
       Parameters
       ----------
       path_to_audiofile: str
       The string is the path to the audio m4a file.
       path_to_split_audio:str
       The path to save the audio segmented according to the duration.
       
       Returns
       -------
       result: str
       path to audio segment for a given audio file
       """    
    if not os.path.exists(path_to_split_audio):
        os.makedirs(path_to_split_audio)

    split_path = os.path.split(path_to_audiofile)
    # mp3_path = path_to_audiofile[:-4]+".mp3"
    # mp4_pydub = AudioSegment.from_file(path_to_audiofile, format="mp4")
    # mp4_pydub.export(mp3_path, format="mp3")
    sample_audio = AudioSegment.from_mp3(path_to_audiofile)

    # if sample_audio.frame_rate < 8000:
    #     print "Frame rate too low"
    if sample_audio.frame_rate > 16000:
        sample_audio = sample_audio.set_frame_rate(16000)

    limit_sec = 59*1000
    no_of_splits = int(np.ceil(sample_audio.duration_seconds/59))
    
    for split in range(no_of_splits):
        batch = sample_audio[limit_sec*(split):min(limit_sec*(split+1),len(sample_audio))]

        path = os.path.join(path_to_split_audio,split_path[1][:-4])
        if not os.path.exists(path):
            os.mkdir(path)
            batch.export(path+"/"+split_path[1][:-4]+"_"+str(split)+".mp3", format="mp3")
        else:
            batch.export(path+"/"+split_path[1][:-4]+"_"+str(split)+".mp3", format="mp3")
    return path


def getTexts(AUDIO_FILE,lan):
    """Takes in an audiofile and return text of the given audio input
         Parameters
         ----------
         AUDIO_FILE: str
         path to audio file in mp3 format
         lan:str
         This attribute will set the language of the recognition
         
         Returns
         -------
         result: text
         Recognized text for the audio snippet
           """
    temp=AUDIO_FILE[:-4]
    AudioSegment.from_file('%s.mp3'%(temp)).export('%s.wav'%(temp), format = 'wav')
    #AudioSegment.from_mp3('%s.mp3'%(temp)).export('%s.wav'%(temp), format="wav")
    AUDIO_FILE = '%s.wav'%(temp)

    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
      audio = r.record(source)  # read the entire audio file
    #mp3_dct[temp]=r.recognize_google(audio,key="AIzaSyCZhrh8FXctpN1SXqXKXx17zn8ydDG8HZE",show_all=True,language="en-IN")
    #mp3_dct[temp]
    
    try:
        audio_text=r.recognize_google_cloud(audio, credentials_json=GOOGLE_APPLICATION_CREDENTIALS,language=lan) #lan="kn-IN"
  #         logging.info('........'+ str(AUDIO_FILE)+' translated')
  #         logging.info('........Done')

    except sr.UnknownValueError:
  #         logging.info('........Google api: UnknownValueError for :  '+ str(AUDIO_FILE))
        audio_text=''
    except sr.RequestError as e:
  #         logging.info('........Google api: RequestError for:  '+ str(AUDIO_FILE))
        audio_text=''

    print("-----audio_text:-------", audio_text)
    mp3_dct={"text":audio_text}
    return  mp3_dct

def translate_english(text):
    """Translates a given text into target language
    Parameters
    ----------
      text: str
                    
    Returns
    -------
    result: translated text
    Text translated into a target language, say:en(English)
    """
    translate_client = translate.Client()
    target = 'en'


    translation = translate_client.translate(
      text,target_language=target)

    return translation['translatedText']



def audio_to_text(path_to_audio_split_folder): #loc read if exists then.. else folder then blah blah
    text = "" 
    
    for i in natsorted(os.listdir(path_to_audio_split_folder), reverse=False):
       if i[-4:] == ".mp3":
           print(i)
           try: 
               print("*************")
               text+=getTexts(os.path.join(path_to_audio_split_folder,i),'en-IN')['text']
           except:
               print("&&&&&&&&&&&&&&&")
               continue
          #ascii codec error
          #print type(getTexts(os.path.join(path_to_audio_split_folder,i),'en-IN')['text'])
          #text+=getTexts(os.path.join(path_to_audio_split_folder,i),'en-IN')['text'].encode('utf-8').decode('ascii', 'ignore') #.decode("ascii","ignore")
   
    return text


def text_conversion(path_to_audio_split_folder, path_to_text_folder): #write to audio_to_text
    logging.info("TC_START for audio split folder: {0}". format(path_to_audio_split_folder))
    split_path = os.path.split(path_to_audio_split_folder)
    if not os.path.exists(path_to_text_folder):
        os.mkdir(path_to_text_folder)

    path_ = os.path.join(path_to_text_folder, "enriched_text.txt")
    print("type of audio to text: ", type(audio_to_text(path_to_audio_split_folder)))
    with open(path_, "w") as myTextFile:
        #myTextFile.write(audio_to_text(path_to_audio_split_folder).encode("utf-8"))
        myTextFile.write(audio_to_text(path_to_audio_split_folder))
       
        
    logging.info("TC_TRANSCRIPT_PATH_CREATED: {0}".format(path_))
    logging.info("TC_STOP for audio split folder: {0}". format(path_to_audio_split_folder))
    return path_

#for json file, get text by key
#input json file, key(string)
def getText_json(jdata,key):
    sdata=json.dumps(jdata)
    regex=key+'": "(.*?)"'
    text_list=([sdata[(m.start(0)+len(key)+4):m.end(0)-1] for m in re.finditer(key+'": "(.*?)"', sdata)])
    return text_list


def ecar_zip_file_processing(path_to_id):
    
    audio_text = ''
    print(path_to_id)
    id = os.path.split(path_to_id)[0]
    logging.info("EZFP_START for id: {0}".format(id))
    # audio processing
    try:
        audio_names = findFiles(os.path.join(path_to_id, 'assets'), ['mp3'])
        print('...detected ' + str(len(audio_names)) + ' audio files')
        if audio_names:
            audiofile_count = 0
            text = ''

            for file in audio_names:
                print(file)
                try:
                    text += getTexts(file, 'en-IN')['text']
                    audiofile_count += 1
                except:
                    continue
                audio_text = text

    except:
        print("The audio file cannot be processed")
        pass

    # image processing
    try:
        image_names = findFiles(os.path.join(path_to_id, 'assets'), ['png', 'gif', 'jpg'])
        print('...detected ' + str(len(image_names)) + ' image files')
        imagefile_count = 0
        text = ''
        logging.info('...image file processing started')
        for file in image_names:
            try:
                text += getImgTags(file)['text']
                imagefile_count += 1

            except:
                  print('........ Error: could not process file')
        text = list(str(text.lower()).split("\n"))
        image_text = ' '.join(list(set(text)))
        print("Image_text: ", image_text)
    except:
        print('...no image file processed')
        logging.info(".....no image file processed")
        image_text = ""


    # mp4 processing
    try:
        video_names = findFiles(os.path.join(path_to_id, 'assets'), ['mp4'])
        print('...detected ' + str(len(video_names)) + ' video files')

        logging.info('...video file processing started')
        video_text = ""
        for file in video_names:
            print(file)
            # ffmpy wrapper to convert mp4 to mp3:
            ff = ffmpy.FFmpeg(
                inputs={file: None},
                outputs={os.path.join(file[:-4] + ".mp3"): '-vn -ar 44100 -ac 2 -ab 192 -f mp3'}
            )
            ff.run()
            if os.path.exists(os.path.join(path_to_id, "assets", file[:-4] + ".mp3")):
                path_to_audio = os.path.join(path_to_id, "assets", file[:-4] + ".mp3")
                path_to_split_audio_download = os.path.join(path_to_id, "assets", "audio_split")

                path_to_split = audio_split(path_to_audio, path_to_split_audio_download)
                logging.info("Path to audio file split is {0}".format(path_to_split))
                video_text+=audio_to_text(path_to_split)
    except:
        print(".....No video files detected")
        logging.info("....No video files detected")
        video_text = ""
        
    try:
        pdf_names = findFiles(os.path.join(path_to_id, 'assets'), ['pdf'])
        print('...detected ' + str(len(pdf_names)) + ' pdf files')
        logging.info(".....detected {0} pdf files".format(str(len(pdf_names))))

        pdf_text = " "
        for pdf_files in pdf_names:
            text = ""
            r =  open(pdf_files, 'rb')  
            read_pdf = PdfFileReader(r)
            number_of_pages = read_pdf.getNumPages()
        #iterate through every page in the PDF
            for i in range(number_of_pages):
                page = read_pdf.getPage(i)
                page_content = page.extractText()
                text+=page_content
            #processed_txt = word_proc(text)
            # text = text.encode("utf-8").decode("ascii", "ignore")
            # text = str(text)
            processed_txt = cleantext(text)# include removing page
            text = ''.join([i for i in processed_txt if not i.isdigit()])
            text = ' '.join(text.split())# check string operation
            pdf_text+=text

    except:
        print(".....No pdf files detected")
        logging.info("....No pdf files detected")
        pdf_text = " "


    # ecml and json file processing:
    if os.path.exists(os.path.join(path_to_id,"index.ecml")) or os.path.exists(
            os.path.join(path_to_id, "manifest.json")):
        all_text = ''
        try:
            xmldoc = ET.parse(os.path.join(path_to_id, "index.ecml"))
            all_text = ''

            xmldoc = minidom.parse(ecml_file)
            print('...File type detected as ecml')
            count_text = 0
            if any(i.tag in ['text', 'htext'] for i in xmldoc.iter()):
                for elem in xmldoc.iter():
                    if ((elem.tag == 'text') or (elem.tag == 'htext')):
                        count_text += 1
                        try:
                            text = str(elem.text).strip() + ' '
                        except:
                            text = elem.text
                        all_text += " " + text

            else:
                print('........ No text or htext detected')
                text = " "
            all_text += " " + audio_text + image_text + video_text
            all_text = all_text.encode("utf-8").decode("ascii", "ignore")

        except:
            # get json text and translate
            try:
                print('...File type detected as json')
                json_data = open(os.path.join(path_to_id, "manifest.json"))
                jdata = json.load(json_data)
                jtext = getText_json(jdata, '__text')

              # jfeedback=getText_json(jdata,'feedback')# would repeat for mcq type


                for text in jtext:  # +jfeedback
                    try:
                        all_text += " " + text
                    except:
                        print('........ Unable to extract json text')
                        text = ""
                all_text += " " + audio_text + image_text + video_text + pdf_text
                all_text = all_text.encode("utf-8").decode("ascii", "ignore")


            except:
                print('...File type neither ecml or json. Skipped.')



    else:
        all_text = audio_text + image_text + video_text + pdf_text
        all_text = all_text.encode("utf-8").decode("ascii", "ignore")
    if not os.path.exists(path_to_id):
        os.mkdir(path_to_id)
    with open(os.path.join(path_to_id, "enriched_text.txt"), 'w') as text_file:
        text_file.write(all_text)
    return os.path.join(path_to_id, "enriched_text.txt")


# creating process_file function
def content_to_text_conversion(url, type_of_url, id_name, path_to_save):#, pathTotextCorpus):
    print(type_of_url)
    logging.info("CTC_START")
    assert type(url) == str
    assert type_of_url == "youtube" or type_of_url == "youtube_embed" or type_of_url == "pdf" or type_of_url == "ekstep_archived" or type_of_url == "unknown" or type_of_url == "zip"
    youtube_bug = []
    pdf_bug = []
    print("path_to_save: {0}".format(path_to_save))
    path_to_id = os.path.join(path_to_save, id_name)
    location = [os.path.join(path_to_id,folder) for folder in ['assets','data','items']]
    
    path_to_audio_download = os.path.join(path_to_id, "assets")
    path_to_split_audio_download = os.path.join(path_to_id, "assets", "audio_split")
    #path_to_transcripts = os.path.join(path_to_id,"enriched_text.txt")
    # if not os.path.exists(pathTotextCorpus):
    #   os.makedirs(pathTotextCorpus)
    if (type_of_url == "youtube" ) or (type_of_url == "youtube_embed" ): 
        logging.info("CTC_YOUTUBE_URL_START for url: {0} and id_name: {1}".format(url, id_name))
        # try:
        for loc in location:
            if not os.path.exists(loc):
                os.makedirs(loc)

        path_to_audio = url_to_audio_extraction(url, path_to_audio_download, type_of_url)
        logging.info("Path to audio file is {0}".format(path_to_audio))

        path_to_split = audio_split(path_to_audio, path_to_split_audio_download)
        logging.info("Path to audio file split is {0}".format(path_to_split))
        path_to_transcript = text_conversion(path_to_split, path_to_id)
        logging.info("Path to audio transcript is {0}".format(path_to_transcript))
        logging.info("-----------------------------------------------------------")
        return path_to_transcript
        # except:
        #       logging.info("Exception raised for youtube url!!")
        #       print "Exception raised for youtube url!!"
        logging.info("CTC_YOUTUBE_URL_END for url: {0} and id_name: {1}".format(url, id_name))    
      
      
    if type_of_url == "pdf":
        try:
            logging.info("CTC_PDF_URL_START for url: {0} and id_name: {1}".format(url, id_name))
            text = ""
            r = requests.get(url)  
            f = io.BytesIO(r.content)
            read_pdf = PdfFileReader(f)
            number_of_pages = read_pdf.getNumPages()
            #iterate through every page in the PDF
            for i in range(number_of_pages):
                page = read_pdf.getPage(i)
                page_content = page.extractText()
                text+=page_content
                #processed_txt = word_proc(text)
            #text = text.encode("utf-8").decode("ascii", "ignore")
            #text = str(text)
            processed_txt = cleantext(text)# include removing page
            result = ''.join([i for i in processed_txt if not i.isdigit()])
            result = ' '.join(result.split())# check string operation
            
            for loc in location:
                if not os.path.exists(loc):
                    os.makedirs(loc)
            
            pdfFile_path = os.path.join(path_to_id, 'enriched_text.txt')
            with open(pdfFile_path, 'w') as local_file:
                 local_file.write(result)
            return pdfFile_path
            logging.info("CTC_PDF_TRANSCRIPT_PATH_CREATED: {0}".format(pdfFile_path))
            logging.info("Path to pdf transcript is {0}".format(pdfFile_path))
        
        except:
            logging.info("Processing of PDF failed {0}".format(url))
        logging.info("CTC_PDF_URL_END for url: {0} and id_name: {1}".format(url, id_name))
            
    if type_of_url == "ekstep_archived" or type_of_url == "zip":
       try:
           logging.info("CTC_ECAR_URL_START for url: {0} and id_name: {1}".format(url, id_name))
           download_dir = os.path.join(path_to_save,'temp'+id_name)
           # url = unicode(url)
           status = downloadZipFile(url, download_dir)
           if status:
              unzip_files(download_dir)
              
              copy_main_folders_new(path_to_save, id_name)
           if os.path.exists(path_to_id):
              path_to_transcript = ecar_zip_file_processing(path_to_id)
              logging.info("Path to transcript for ecar url: {0}".format(path_to_transcript))
       except:
              logging.info("Skipped url: {0}".format(url))
       logging.info("CTC_ECAR_URL_END for url: {0} and id_name: {1}".format(url, id_name))



def profanityFilter(words, bad_word_corpus):
  #words is a list of words that needs to be filtered
  # bad_word_corpus is a file location of words to be removed
  
    try:
      block_words_list=list(pd.read_csv(bad_word_corpus, header=None)[0])
      
      cleaned_list=[i for i in words if not any(x in block_words_list for x in i.split(" "))] 
      return cleaned_list
    
    except IOError:
      print('An error occurred trying to read the corpus.')



#general cleaning code snippet------> Utils
def clean_text_tokens(text):
    #text = text.encode("utf-8").decode("ascii", "ignore")
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token.lower() not in stopwords]
    tokens = [token for token in tokens if token not in pun_list]
    tokens = [re.sub('[0-9\.\W_]', '', token) for token in tokens]
    tokens = [token.lower() for token in tokens]#/*
    tokens = [token for token in tokens if len(token)>1]
    tokens = [token for token in tokens if token]
     # Lowercase all words (default_stopwords are lowercase too)
    tokens = [token.lower() for token in tokens]
     
    
    return tokens



def pafy_text_tokens(path_to_text_file, path_to_pafy_text_tokens_folder):
    split_path = os.path.split(path_to_text_file)
    text = open(path_to_text_file, "r")
    text_file = text.read()
    text_list = clean_text_tokens(text_file)
    if not os.path.exists(path_to_pafy_text_tokens_folder):
      os.mkdir(path_to_pafy_text_tokens_folder)
      os.chdir(path_to_pafy_text_tokens_folder)
    else:
      os.chdir(path_to_pafy_text_tokens_folder)
    path = os.path.join(path_to_pafy_text_tokens_folder, "keywords.csv")
    with open(path, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(['KEYWORDS'])
        for i in text_list:
            wr.writerows([[i]])


    return path

def tagme_text(text):
    url = "https://tagme.d4science.org/tagme/tag"

    querystring = {"lang":"en","gcube-token":"1e1f2881-62ec-4b3e-9036-9efe89347991-843339462","text":text}

    headers = {
    'gcube-token': "1e1f2881-62ec-4b3e-9036-9efe89347991-843339462",
    'cache-control': "no-cache",
    'postman-token': "98279373-78af-196e-c040-46238512c338"
    }

    response = requests.request("GET", url, headers=headers, params=querystring).json()
    df = pd.DataFrame(response['annotations']) 
    return df 

def get_tagme_longtext(path_to_text, path_to_tagme):
    file_ =open(path_to_text, "r")
    text = file_.readline()
    text = text.encode('utf-8').decode('ascii', 'ignore') 
    words = text.split(" ")
    index_count=0
    window_len=700
    response_list=[]
    split_path = os.path.split(path_to_text)
    if not os.path.exists(path_to_tagme):
        os.makedirs(path_to_tagme)
    while index_count<len(words) :
        text=' '.join(words[index_count:min((index_count+window_len-1),len(words))])
        index_count += window_len
        response_list.append(tagme_text(text))
        response_df = pd.concat(response_list) 
        response_df=response_df.drop_duplicates('spot')
        response_df.reset_index(drop=True,inplace=True) 
        cleaned_keyword_list = [str(x).lower() for x in list(set(response_df['spot'])) if str(x) != 'nan']                   
        cleaned_keyword_list = clean_string_list(cleaned_keyword_list)
        unique_cleaned_keyword_list = list(set(cleaned_keyword_list))
        spot = pd.DataFrame(unique_cleaned_keyword_list, columns = ['KEYWORDS'])
        path = os.path.join(path_to_tagme, "keywords.csv")
        spot.to_csv(path, index=False, encoding='utf-8')
    return path


def tagme_taxonomy_intersection_keywords(taxonomy_keywords_set, path_to_tagme_keys, path_to_tagme_taxonomy_folder):
    try:
        split_path = os.path.split(path_to_tagme_keys)
        tagme_spots = pd.read_csv(path_to_tagme_keys, sep=',',index_col=None)
        spots = [i.lower() for i in list(tagme_spots['KEYWORDS'])] 
        
        common_words = list(set(taxonomy_keywords_set)&set(spots))
        
        if not os.path.exists(path_to_tagme_taxonomy_folder):
           os.makedirs(path_to_tagme_taxonomy_folder)
          

        path = os.path.join(path_to_tagme_taxonomy_folder, 'keywords.csv')
        with open(path, 'w') as myfile:
          wr = csv.writer(myfile)
          wr.writerow(['KEYWORDS'])
          for i in common_words:
            wr.writerows([[i]])
        return path
    except:
        logging.info("Keywords cannot be extracted")

# #? dont forget to put back self as an argument
# def keyword_extraction_parallel(dir, path_to_texts, taxonomy, extract_keywords, method, filter_criteria):
#         print("*******dir*********:", dir)
#         print("***Extract keywords***:", extract_keywords)
#         print("***Filter criteria:***", filter_criteria)
#         path_to_cid_transcript = os.path.join(path_to_texts, dir, "enriched_text.txt")
#         keywords = os.path.join(path_to_texts, dir, "keywords")
#         path_to_text_tokens = os.path.join(keywords, "text_tokens")
#         path_to_tagme_tokens = os.path.join(keywords, "tagme_tokens")
#         path_to_tagme_taxonomy_intersection = os.path.join(keywords, "tagme_taxonomy_tokens")

#         if os.path.isfile(path_to_cid_transcript):
#             logging.info("Transcript present for cid: {0}".format(dir))
#             try:

#                 # text_file = os.listdir(path_to_cid_transcript)[0]
#                 if os.path.getsize(path_to_cid_transcript) > 0:
#                     print("Path to transcripts ", path_to_cid_transcript)
#                     logging.info("Running keyword extraction for {0}".format(path_to_cid_transcript))
#                     logging.info("---------------------------------------------")

#                     if extract_keywords == True and filter_criteria == "none":
#                         logging.info("Tagme keyword extraction is running for {0}".format(path_to_cid_transcript))
#                         path_to_pafy_tagme_tokens = get_tagme_longtext(path_to_cid_transcript, path_to_tagme_tokens)
#                         logging.info("Path to tagme tokens is {0}".format(path_to_pafy_tagme_tokens))

#                     elif extract_keywords == False:
#                         logging.info("Text tokens extraction running for {0}".format(path_to_cid_transcript))
#                         path_to_pafy_text_tokens = pafy_text_tokens(path_to_cid_transcript, path_to_text_tokens)
#                         logging.info("Path to text tokens is {0}".format(path_to_pafy_text_tokens))

#                     elif extract_keywords == True and filter_criteria == "taxonomy_keywords":
#                         logging.info("Tagme intersection taxonomy keyword extraction is running for {0}".format(
#                             path_to_cid_transcript))
#                         revised_content_df = pd.read_csv(taxonomy, sep=",", index_col=None)
#                         clean_keywords = map(get_words, list(revised_content_df["Keywords"]))
#                         clean_keywords = map(clean_string_list, clean_keywords)
#                         flat_list = [item for sublist in list(clean_keywords) for item in
#                                      sublist]
#                         taxonomy_keywords_set = set([cleantext(i) for i in flat_list])
#                         path_to_pafy_tagme_tokens = get_tagme_longtext(path_to_cid_transcript, path_to_tagme_tokens)
#                         path_to_tagme_intersect_tax = tagme_taxonomy_intersection_keywords(taxonomy_keywords_set,
#                                                                                            path_to_pafy_tagme_tokens,
#                                                                                            path_to_tagme_taxonomy_intersection)
#                         logging.info \
#                             ("Path to tagme taxonomy intersection tokens is {0}".format(path_to_tagme_intersect_tax))

#                     else:
#                         logging.info("Invalid argument provided")


#                 else:
#                     logging.info("The text file {0} has no contents".format(path_to_cid_transcript))
#                     print("The text file {0} has no contents".format(path_to_cid_transcript))

#             except:
#                 print("Raise exception for {0} ".format(path_to_cid_transcript))
#                 logging.info("Raise exception for {0} ".format(path_to_cid_transcript))
#         else:
#             logging.info("Transcripts doesnt exist for {0}".format(path_to_cid_transcript))
#             print("Transcripts doesnt exist for {0}".format(path_to_cid_transcript))

#         return path_to_texts


def clean_string_list(x_list):
     #print x_list
     #map object changed to list
     x_list=list(map((str.lower),x_list))
     x_clean=[i.lstrip() for i in x_list]
     x_clean=[i.rstrip() for i in x_clean]
     x_clean=list(filter(None, x_clean))
     return x_clean


def get_words(x):
    
    if str(x)!='nan':
     x=x.split(', ')        
     return x
    else:
     return ""

def custom_listPreProc(key_list, preproc, DELIMITTER):
    key_list=[clean_string_list(x) for x in key_list]
    key_list_clean=[]
    for x in key_list: 
        x=[word_proc(i) for i in x]
        if preproc == 'stem_lem':
            key_list_clean.append(stem_lem((x),DELIMITTER))
        else:
            print("unknown preproc")
            return
    return key_list_clean


def stem_lem(keyword_list, DELIMITTER):
    wordnet_lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    keyword_list=[item for item in keyword_list]
    keyword_list=[i.split(DELIMITTER) for i in keyword_list]
    lemma_ls_1 = [[wordnet_lemmatizer.lemmatize(item, pos="n") for item in words] for words in keyword_list]
    lemma_ls_2 = [[wordnet_lemmatizer.lemmatize(item, pos="v") for item in words] for words in lemma_ls_1]
    lemma_ls_3 = [[wordnet_lemmatizer.lemmatize(item, pos="a") for item in words] for words in lemma_ls_2]
    lemma_ls_4 = [[wordnet_lemmatizer.lemmatize(item, pos="r") for item in words] for words in lemma_ls_3]
    stemm_ls = [[stemmer.stem(item) for item in words] for words in lemma_ls_4]


    return [DELIMITTER.join(i) for i in stemm_ls]
  
   
def get_level_keywords(taxonomy_df,level):
    level_keyword_df=[]
    for subject in list(set(taxonomy_df[level])):
        Domain_keywords= list(taxonomy_df.loc[taxonomy_df[level]==subject,'Keywords'])
        unique_keywords=[ind for sublist in Domain_keywords for ind in sublist]
     
        level_keyword_df.append({level:subject,'Keywords':unique_keywords})
    level_keyword_df=pd.DataFrame(level_keyword_df)
    return level_keyword_df

def getGradedigits(class_x):
    for i in ["Class","[","]"," ","class","Grade","grade"]:
        class_x=class_x.replace(i,"")
    return class_x


def match_str_list(list1,list2):
     intersection= 1.0*(len(set(list1)&set(list2)))
     union=1.0*(len(set(list1+list2)))
     if union !=0:
       jaccard_index=intersection/union
     else:
       jaccard_index=0
     try:
       cosine_similarity=intersection/(len(set(list1))*len(set(list2)))
     except:
       cosine_similarity=0
     if len(set(list1))!=0:
       match_percent_l1=float(intersection)/len(set(list1))
     else:
       match_percent_l1=0
     if len(set(list2))!=0:
       match_percent_l2=float(intersection)/len(set(list2))
     else:
       match_percent_l2=0
     return {'intersection':intersection,'union':union,'jaccard_index':jaccard_index,'cosine_similarity':cosine_similarity,'match_percent_l1':match_percent_l1,'match_percent_l2':match_percent_l2}


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def getMinEditDist(l1,l2):
     dist_matrix=[]
     for row_word in l1:
       row_vec=[]
       for col_word in l2:
           row_vec.append(1.0*Levenshtein.distance(row_word.decode('utf-8'), col_word.decode('utf-8'))/max(len(row_word),len(col_word)))
       dist_matrix.append(row_vec)  
     dist=(pd.DataFrame(dist_matrix,columns=l2, index=l1))
     return dist


def getWordlistEMD(doc1,doc2):
     keyword_union=doc1+doc2

     vectorizer = CountVectorizer()
     X = vectorizer.fit(keyword_union)
     hist1=X.transform([', '.join(doc1)]).todense()
     hist2=X.transform([', '.join(doc2)]).todense()
     h1=normalize(np.array(np.array(hist1).reshape(-1,)).astype('float64'))
     h2=normalize(np.array(np.array(hist2).reshape(-1,)).astype('float64'))
     keyword_union_set=(X.vocabulary_).keys()
     med_dist=(getMinEditDist(keyword_union_set,keyword_union_set).values).astype('float64')
     med_dist = med_dist.copy(order='C')
     dist=emd(h1, h2,med_dist)
     return dist


def getHeatMap(x):
     data=go.Heatmap(
       y = list(x.columns),
       x =list(x.index),
       z = x.T.values.tolist(),
       colorscale = 'YIGnBu'
     )
     fig = go.Figure(data=[data])

     iplot(fig)


def dictionary_merge(dict_ls):

    keys_ls = [dictionary.keys() for dictionary in dict_ls]
    keys_ls = set([elem for sublist in keys_ls for elem in sublist])  
    dict_all = {keys: pd.DataFrame() for keys in (keys_ls)} 
    for dictionary in dict_ls: 
        for keys in dictionary.keys():
            dict_all[keys] = dict_all[keys].append(dictionary[keys])  
    return dict_all 
        
   
def get_sorted_list(x,order): #order=0-decreasing(Jaccard), 1-increasing(MED,EMD)
     x_df=pd.DataFrame(x)
     return list(x_df.sort_values(by=list(x_df.columns), ascending=order).index)


def getEvalMatrix(Content_topic_predicted,Content_topic_true,level, sort_order,window_len):
    Content_topic_predicted=Content_topic_predicted.T.apply(func=lambda x:get_sorted_list(x,sort_order),axis=0).T
    Content_topic_predicted.columns=range(Content_topic_predicted.shape[1])
    
    window=range(1,window_len+1)
    percent_list=[]
    for ind in window:
        count=0
        for cid in Content_topic_predicted.index:
#            try:
             if (Content_topic_true.loc[cid,level]) in list(Content_topic_predicted.loc[cid][0:ind]):
                    count+=1
#            except:
#               print str(cid)+" metadata not available "
        percent_list.append(count*100.0/len(Content_topic_predicted.index))

    return pd.DataFrame(percent_list, index=window, columns=["percent"])


def getEval_agg(dist_all,truetopic_all, sort_order):
  
    distance_consolidate_list=[]
    true_topic_list=[]
    distance_list=dist_all
    for subject in truetopic_all.keys():
      Content_topic_predicted=distance_list[subject].T.apply(func=lambda x:get_sorted_list(x,sort_order),axis=0).T
      for i in range(Content_topic_predicted.shape[0]):
        distance_consolidate_list.append(list(Content_topic_predicted.iloc[i]))
        true_topic_list.append(truetopic_all[subject].iloc[i][0])
     
    window=[0]*16
    for i in range(0,len(window)):
      count=0
      for content in range(len(distance_consolidate_list)):
        if true_topic_list[content] in distance_consolidate_list[content][0:min(len(distance_consolidate_list[content]),i+1)]:
          count+=1
      window[i]=100.0*count/len(distance_consolidate_list)
    return pd.DataFrame(window,columns=['Percent'])


# def getEvalMatrix_withagg(dist_dict,truetopic_all, sort_order):
#     predicted_topic_list=[]
#     true_topic_list=[]
#     for subject in truetopic_all.keys():
#         Content_topic_predicted=dist_dict[subject].T.apply(func=lambda x:get_sorted_list(x,sort_order),axis=0).T
#         for i in range(Content_topic_predicted.shape[0]):
#             predicted_topic_list.append(list(Content_topic_predicted.iloc[i]))
#             true_topic_list.append(truetopic_all[subject].iloc[i][0])
  
#     window=[0]*16
#     for i in range(0,len(window)):
#         count=0
#         for content in range(len(predicted_topic_list)):
#             if true_topic_list[content] in predicted_topic_list[content][0:min(len(predicted_topic_list[content]),i+1)]:
#                 count+=1
#         window[i]=100.0*count/len(jaccard_consolidated_list)
#     return pd.DataFrame(window) 

def get_prediction(dist_df,sort_order,number_of_pred):
    mapped_df=dist_df.T.apply(func=lambda x:get_sorted_list(x,sort_order),axis=0).T
    mapped_df.columns=range(1,mapped_df.shape[1]+1)
    return mapped_df.iloc[:,0:number_of_pred]

  
def keywordAddition_contentMetadata(url, content_metadata_path, path_to_datasource):
    Content_metadata=pd.read_csv(content_metadata_path, index_col=0)
    content_id = clean_url(url)+".csv"
    print(content_id)
    Content_metadata = Content_metadata[Content_metadata["new_cid"] == content_id]
    
    content_keywords_list=[]
    extracted_keyword_df=pd.read_csv(os.path.join(path_to_datasource), keep_default_na=False)#tagme_keywords/
    extracted_keys=list(extracted_keyword_df['KEYWORDS'])
    extracted_keys=clean_string_list(extracted_keys)
    content_keywords_list.append(extracted_keys)
    Content_metadata['Content_keywords']=content_keywords_list
    return Content_metadata

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
      pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin1') 


def getPhrase(x_list,DELIMITTER):
    #x_list=clean_string_list(x_list)
    x_phrase=[i for i in x_list if DELIMITTER in i]
    x_word=[ item for item in x_list if item not in x_phrase ]
    return x_word,x_phrase


def removeShortWords(mylist, wordlen):
    return [item for item in mylist if len(item)>wordlen]


def WordtoPhraseMatch(wordlist,phraselist,DELIMITTER):
    phrasewords=[item.split(DELIMITTER) for item in phraselist ]
    match_count=0
    partial_match_list=[]
    wordlist_dynamic=wordlist[:]
    for items in phrasewords:
    
        word_count=0
        word_list=[]
        wordlist=wordlist_dynamic[:]
        for word in wordlist:
            
            if word in items:
                word_count+=1
                partial_match_list.append((word,'_'.join(items)))
                wordlist_dynamic.remove(word)
        match_count+=int(bool(word_count))       
    return partial_match_list,match_count


#jaccard_with_phrase
def jaccard_with_phrase(list1,list2):#list1 is taken as content and list2 as tax. Precedence is given to taxonomy matchin
    DELIMITTER="_"
    intersection_words=[]
    list1=list(set(list1))
    list2=list(set(list2))
    list1=removeShortWords(list1,0)
    list2=removeShortWords(list2,0)
    list1_words,list1_phrases=getPhrase(list1,DELIMITTER)
    list2_words,list2_phrases=getPhrase(list2,DELIMITTER)
    list1_len=len(list1_words)+len(list1_phrases)
    list2_len=len(list2_words)+len(list2_phrases)
    intersection=0
    match_count=0
    
    ##count matching words
    exact_word_intersection=list(set(list1_words) & set(list2_words))
    intersection_words.extend([(a,a) for a in exact_word_intersection])
    
    exact_word_match=match_str_list(list1_words,list2_words)['intersection']
    intersection=intersection+exact_word_match
    match_count+=exact_word_match
    #print "intersection1:"+str(intersection)

    ##count matching phrases
    #exact_phrase_intersection=list(set(list1_phrases)&set(list2_phrases))  
    #exact_phrase_length= sum([len(i.split(DELIMITTER)) for i in exact_phrase_intersection ])
    
    #exact_phrase_intersection=[(phrase1,phrase2) for phrase1 in list1_phrases for phrase2 in list2_phrases if((phrase2 in phrase1) or (phrase1 in phrase2) )]
    ### revised for single count of phrase match
    phraselist1=list1_phrases
    phraselist2=list2_phrases
    exact_phrase_intersection=[]
    for phrase1 in phraselist1:
        for phrase2 in phraselist2:
            if((phrase2 in phrase1)  or (phrase1 in phrase2)):
                exact_phrase_intersection.append((phrase1,phrase2) )
                list2_phrases.remove(phrase2)
                break

    ###
    exact_phrase_length=sum([min([(len(j.split(DELIMITTER))) for j in i ]) for i in exact_phrase_intersection])
    
    intersection+=(2.0*exact_phrase_length)
    match_count+=len(exact_phrase_intersection)
    #print "intersection2:"+str(intersection)
    intersection_words.extend(exact_phrase_intersection)  
    
    non_matched_list1_words,non_matched_list2_words=list1_words, list2_words
    non_matched_list1_phrases,non_matched_list2_phrases=list1_phrases,list2_phrases
    
    if exact_word_intersection:
        non_matched_list1_words=[ item for item in list1_words if str(item) not in exact_word_intersection ]
        non_matched_list2_words=[ item for item in list2_words if str(item) not in exact_word_intersection ]
    if exact_phrase_intersection:
        non_matched_list1_phrases=[word for item in exact_phrase_intersection for word in non_matched_list1_phrases if item[0] not in word]   
        non_matched_list2_phrases=[word for item in exact_phrase_intersection for word in non_matched_list2_phrases if item[1] not in word]   
    
    #Content words to taxonomy phrases
    partial_match_list1,count=WordtoPhraseMatch(non_matched_list1_words,non_matched_list2_phrases,DELIMITTER)
    match_count+=count
    
    if partial_match_list1:
        non_matched_list1_words=[word for item in partial_match_list1 for word in non_matched_list1_words if item[0] not in word]
        non_matched_list2_phrases=[word for item in partial_match_list1 for word in non_matched_list2_phrases if item[1] not in word]   
    
    intersection=intersection+len(partial_match_list1)
    intersection_words.extend(partial_match_list1)

    #Content phrase to taxonomy words
    partial_match_list2,count=WordtoPhraseMatch(non_matched_list2_words,non_matched_list1_phrases,DELIMITTER)
    match_count+=count
      
    
    non_matched_list2_words=[item[0] for item in partial_match_list2 if item[0] not in non_matched_list1_phrases]
    intersection=intersection+len(partial_match_list2)
    
    intersection_words.extend(partial_match_list2)
    intersection_words=[el for el in intersection_words if el!=[]]

    if (((len(list2))!=0) & ((len(list1)+len(list2)-match_count)!=0)):
        return {'jaccard1':float(intersection)/float(len(list1)+len(list2)-match_count),'jaccard2':float(intersection)/float(len(list2)),'word_intersection':intersection_words}
    elif ((len(list1)+len(list2)-match_count)==0):

        return {'jaccard1':0,'jaccard2':float(intersection)/float(len(list2)),'word_intersection':intersection_words}
    elif (len(list2)==0):
        return {'jaccard1':float(intersection)/float(len(list1)+len(list2)-match_count),'jaccard2':0,'word_intersection':intersection_words}

#jaccard_with_phrase(domain_content_df['Content_keywords'][row_ind],level_domain_taxonomy_df['Keywords'][col_ind])






