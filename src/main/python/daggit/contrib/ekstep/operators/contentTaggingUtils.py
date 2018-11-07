import io
import requests
import os
import logging

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
import pickle
import youtube_dl

import mutagen.mp3 as mp3
import speech_recognition as sr
import pandas as pd
import numpy as np
from PyPDF2 import PdfFileReader
import googleapiclient.discovery
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import time

from google.cloud import translate
from google.cloud import vision
from google.cloud.vision import types
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


def downloadZipFile(url, directory):
    """
    Multimedia Content are stored in cloud in ecar or zip format.
    This function downloads a zip file pointed by url location.
    The user is expected to have access to the file pointed by url.
    The extracted file is available in location specified by directory.
  
    Parameters
    ----------
    url: str
    A valid url pointing to ziped Content location on cloud
    
    directory: str
    path to download and extract the zip file
  
    Returns
    -------
    bool: Status of download. True: succesful download  and False: for unsuccesful download
    """
    r = requests.get(url)
    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(directory)
        return r.ok
    except:
        return False


def findFiles(directory, substrings):
    """
    Accio!!
    For a given directory, the function looks for any occurance of a particular file type mentioned by the substrings parameter.

    Parameters
    ----------
    directory: str
    path to a folder

    substrings: list
    an array of extentions to be searched within the directory. ex: png, webm, mp4

    Returns
    -------
    list: list of paths to detected files
    """
    ls = []
    if type(directory) == str and type(substrings) == list:
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
    zip_list=findFiles(directory,['.zip'])
    bugs={}
    for zip_file in zip_list:
      try:
        with zipfile.ZipFile(zip_file, 'r') as z:
          z.extractall(directory)
        os.remove(zip_file)
      except:
        bugs.append(zip_file)


def ekstep_ecar_unzip(download_location, copy_location):
    """
    This function unzips an ecar file(ekstep file format) and parses all the subfolder.
    All the files are copied into one of 'assets','data','items' folder(same name as in downloaded folder is maintained) based on its location in the downloaded folder.
    ==========
    arguments:
        download_location: A location in the disk where ekstep ecar resource file in  downloaded
        copy_location: A disk location where the ecar is unwrapped
    """
    assert type(download_location)==str
    assert type(copy_location)==str
    if not os.path.exists(copy_location):
        os.makedirs(copy_location)
    if not os.path.exists(path):
        print("path:", path)
        os.makedirs(path)
    location=[os.path.join(copy_location,folder) for folder in ['assets','data','items']]

    for loc in location:
        if not os.path.exists(loc):
            os.makedirs(loc)
    for subfolder in os.listdir(os.path.join(download_location)):
        if os.path.isdir(os.path.join(download_location, subfolder)) and len(os.listdir(os.path.join(download_location, subfolder)))>0:
            for file in os.listdir(os.path.join(download_location, subfolder)):
                shutil.copy(os.path.join(download_location, subfolder, file) , os.path.join(copy_location, "assets"))
        else:
            shutil.copy(os.path.join(download_location, subfolder) , copy_location)


def download_from_downloadUrl(url_to_download, path_to_folder, file_name):
    download_dir = os.path.join(path_to_folder,'temp'+file_name)
    status = downloadZipFile(url_to_download, download_dir)
    if status:
        unzip_files(download_dir)
        ekstep_ecar_unzip(download_dir, os.path.join(path_to_folder,file_name))
        shutil.rmtree(download_dir)
        path_to_file = os.path.join(path_to_folder, file_name)
        return path_to_file


def df_feature_check(df, mandatory_fields):
    """
    Check if columns are present in dataframe

    Parameters
    ----------
    df: dataframe
    DataFrame that needs to be checked

    mandatory_fields: list
    list of column names

    Returns
    -------
    result: bool
    True if all columns are present. False if not all columns are present
    """
    check = [0 if elem in list(df.columns) else 1 for elem in mandatory_fields]
    if sum(check) > 0:
        return False
    else:
        return True


def cleantext(text):
    """
    Custom function to clean enriched text

    Parameters
    ----------
    text: str
    Enriched text from Ekstep Content.

    Returns
    -------
    text: str
    Cleaned text
    """
    replace_char=["[","]","u'","None","Thank you","-","(",")","#","Done" , ">" ,"<","-" ,"|","/","\"","Hint","\n","'"]
    for l in replace_char:
        text = text.replace(l, "")
    text = re.sub(' +', ' ', text)
    return text


def clean_text_tokens(text):
    """
    A custom preprocessor to tokenise and clean a text. Used in Content enrichment pipeline.
    Process:
    - tokenise string using nltk word_tokenize()
    - Remove stopwords
    - Remove punctuations in words
    - Remove digits and whitespaces
    - Convert all words to lowercase
    - Remove words of length 1
    - Remove nan or empty string

    Parameters
    ----------
    text: str
    The string to be tokenised

    Returns
    -------
    token: list
    list of cleaned tokenised words
    """
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token.lower() not in stopwords]
    tokens = [token for token in tokens if token not in pun_list]
    tokens = [re.sub('[0-9\.\W_]', '', token) for token in tokens]
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if len(token) > 1]
    tokens = [token for token in tokens if token]
    return token


def strip_word(word, delimitter):
    """
    Replace  punctuations from string.punctuation and space in a word with a DELIMITTER

    Parameters
    ----------
    word: str
    Typically a word whose punctuations and space are removed

    DELIMITTER: str
    string to replace punctuations 
  
    Returns
    -------
    result: str 
    Processed string
    """
    delimitters = [" ", ",", "_", "-", ".", "/"] + list(set(string.punctuation))
    for lim in delimitters:
        word = word.replace(lim, delimitter)
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
            return "https://www.youtube.com/watch?v=" + fetch_video_id(url.split("embed/")[1])[:11]
        else:
            return "https://www.youtube.com/watch?v=" + youtube_regex_match.group(6)


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
    print(full_text)
    if len(full_text) > 0:
        img_dct = {'text': full_text[0]}   
    else:
        img_dct = {'text': ""}
    return img_dct['text']


def url_to_audio_extraction(url, path):
    logging.info("UTAE_YOUTUBE_URL_START: {0}".format(url))
    if not os.path.exists(path):
        os.makedirs(path)
    cid  = os.path.split(os.path.split(path)[0])[1]
    path_to_audio = os.path.join(path, cid +".mp3")
    print(path_to_audio)
    logging.info("UTAE_YOUTUBE_URL_START: {0}".format(url))
    if not os.path.isfile(path_to_audio):
        os.chdir(path)
        url = embed_youtube_url_validation(url)
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
        return path_to_audio
    else:
        return path_to_audio
    logging.info("UTAE_YOUTUBE_URL_STOP")


def audio_split(path_to_audiofile, path_to_split_audio):
    """
    Takes in an audiofile, split it as per the duration and returns the path to the split
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
    sample_audio = AudioSegment.from_mp3(path_to_audiofile)
    if sample_audio.frame_rate > 16000:
        sample_audio = sample_audio.set_frame_rate(16000)
    limit_sec = 59*1000
    no_of_splits = int(np.ceil(sample_audio.duration_seconds/59))
    for split in range(no_of_splits):
        batch = sample_audio[limit_sec*(split):min(limit_sec*(split+1), len(sample_audio))]
        path = os.path.join(path_to_split_audio,split_path[1][:-4])
        if not os.path.exists(path):
            os.mkdir(path)
            batch.export(path+"/"+split_path[1][:-4]+"_"+str(split)+".mp3", format="mp3")
        else:
            batch.export(path+"/"+split_path[1][:-4]+"_"+str(split)+".mp3", format="mp3")
    return path


def getTexts(AUDIO_FILE, lan):
    """
    Takes in an audiofile and return text of the given audio input
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
    temp = AUDIO_FILE[:-4]
    AudioSegment.from_file('%s.mp3'%(temp)).export('%s.wav'%(temp), format = 'wav')
    #AudioSegment.from_mp3('%s.mp3'%(temp)).export('%s.wav'%(temp), format="wav")
    AUDIO_FILE = '%s.wav'%(temp)

    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
      audio = r.record(source)  # read the entire audio file
    
    try:
        audio_text = r.recognize_google_cloud(audio, credentials_json=GOOGLE_APPLICATION_CREDENTIALS,language=lan) #lan="kn-IN"
  #         logging.info('........'+ str(AUDIO_FILE)+' translated')
  #         logging.info('........Done')

    except sr.UnknownValueError:
  #         logging.info('........Google api: UnknownValueError for :  '+ str(AUDIO_FILE))
        audio_text = ''
    except sr.RequestError as e:
  #         logging.info('........Google api: RequestError for:  '+ str(AUDIO_FILE))
        audio_text = ''

    #print("-----audio_text:-------", audio_text)
    mp3_dct = {"text": audio_text}
    return mp3_dct


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
      text, target_language=target)
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
    return text


def text_conversion(path_to_audio_split_folder, path_to_text_folder): #write to audio_to_text
    logging.info("TC_START for audio split folder: {0}". format(path_to_audio_split_folder))
    split_path = os.path.split(path_to_audio_split_folder)
    if not os.path.exists(path_to_text_folder):
        os.mkdir(path_to_text_folder)

    path_ = os.path.join(path_to_text_folder, "enriched_text.txt")
    print("type of audio to text: ", type(audio_to_text(path_to_audio_split_folder)))
    with open(path_, "w") as myTextFile:
        myTextFile.write(audio_to_text(path_to_audio_split_folder))
       
        
    logging.info("TC_TRANSCRIPT_PATH_CREATED: {0}".format(path_))
    logging.info("TC_STOP for audio split folder: {0}". format(path_to_audio_split_folder))
    return path_


def getText_json(jdata, key):
    sdata=json.dumps(jdata)
    regex=key+'": "(.*?)"'
    text_list=([sdata[(m.start(0)+len(key)+4):m.end(0)-1] for m in re.finditer(key+'": "(.*?)"', sdata)])
    return text_list


def download_to_local(method, url_to_download, path_to_save, id_name):
    logging.info("DTL_START_FOR_URL: {0}".format(url_to_download))
    path_to_id = ""
    if method == "ecar":
        logging.info("DTL_ECAR_URL: {0}".format(url_to_download))
        try:
            logging.info("DTL_FOR_URL".format(url_to_download))
            path_to_id = download_from_downloadUrl(url_to_download, path_to_save, id_name)
        except:
            logging.info("Skipped url: {0}".format(url_to_download))
    
    if method == "youtube":
        logging.info("DTL_YOUTUBE_URL: {0}".format(url_to_download))
        path_to_id = os.path.join(path_to_save, id_name)
        location = [os.path.join(path_to_id,folder) for folder in ['assets','data','items']]

        path_to_audio_download = os.path.join(path_to_id, "assets")
        for loc in location:
            if not os.path.exists(loc):
                os.makedirs(loc)
        path_to_audio = url_to_audio_extraction(url_to_download, path_to_audio_download)
        logging.info("Path to audio file is {0}".format(path_to_audio))
         ##should we returning path_to_audio or path_to_id
    
    if method == "pdf":
        logging.info("DTL_PDF_URL: {0}".format(url_to_download))
        try:
            logging.info("DTL_FOR_URL".format(url_to_download))
            path_to_id = download_from_downloadUrl(url_to_download, path_to_save, id_name)
        except:
            logging.info("Skipped url: {0}".format(url_to_download))
    
    else:
        logging.info("Download not required for url: {0}".format(url_to_download))
    logging.info("DTL_STOP_FOR_URL: {0}".format(url_to_download))
    return path_to_id
 

def video_to_speech(method, path_to_assets): #method can be ffmpeg or youtube_dl
    logging.info('VTS_START')
    video_names = findFiles(path_to_assets, ['mp4'])
    logging.info('...detected {0} video files'.format(str(len(video_names))))
    if method == "ffmpeg" and len(video_names)>0:
        logging.info("VTS_START_FOR_METHOD: {0}".format(method))
        
        for file in video_names:
            # ffmpy wrapper to convert mp4 to mp3:
            ff = ffmpy.FFmpeg(
                inputs={file: None},
                outputs={os.path.join(file[:-4] + ".mp3"): '-vn -ar 44100 -ac 2 -ab 192 -f mp3'}
            )
            ff.run()
            if os.path.exists(os.path.join(path_to_assets, file[:-4] + ".mp3")):
                path_to_audio = os.path.join(path_to_assets, file[:-4] + ".mp3")
                logging.info("VTS_AUDIO_DOWNLOAD_PATH: ".format(path_to_audio))
            else:
                logging.info("mp3 download unsuccessful")
    if method == "none":
        logging.info("No Video content detected")
    logging.info('VTS_STOP')
    return path_to_assets


def speech_to_text(method, path_to_assets): 
    logging.info("STT_START")
    text=""
    if not os.path.exists(path_to_assets): #check for empty string
        logging.info("No audio file detected")
    else:
        audio_names = findFiles(path_to_assets, ['mp3'])
        if method == "googleAT" and len(audio_names)>0:
            for i in audio_names:
                logging.info("STT_AUDIO_FILEPATH: {0}".format(os.path.join(path_to_assets, i)))
                path_to_split = audio_split(os.path.join(path_to_assets, i), 
                                            os.path.join(path_to_assets, "audio_split"))
                logging.info("STT_AUDIO_SPLIT: {0}".format(path_to_split))
                text += audio_to_text(path_to_split)
        elif method  == "none":
            logging.info("STT_NOT_PERFORMED")
        else:
            logging.info("Unknown method given")
    logging.info("STT_STOP")
    return text


def image_to_text(method, path_to_assets):
    logging.info("ITT_START")
    image_text=""
    image_names = findFiles(path_to_assets, ['png', 'gif', 'jpg'])
    if method == "googleVision" and len(image_names) > 0:
        logging.info('...detected {0} video files'.format(str(len(image_names))))
        logging.info('...image file processing started')
        for file in image_names:
            try:
                image_text += getImgTags(file)
            except:
                print('........ Error: could not process file')
        print("Text: ", image_text)        
        text = list(str(image_text.lower()).split("\n"))
        image_text = ' '.join(list(set(text)))
    if method == "none":
        logging.info("ITT_NOT_PERFORMED")
    logging.info("ITT_STOP")
    return image_text


def pdf_to_text(method, path_to_assets, pdf_url):
    text = ""
    logging.info("PTT_START")
    if method == "PyPDF2":
        logging.info("PTT_METHOD: {0}".format(method))
        pdf_names = findFiles(path_to_assets, ['pdf'])
        text = ""
        for j in range(0,len(pdf_names)+1):
            if (len(pdf_names)==0):
                r = requests.get(pdf_url)  
                f = io.BytesIO(r.content)
                read_pdf = PdfFileReader(f)
            elif j<(len(pdf_names)):
                pdf_files=pdf_names[j]
                text = ""
                f = open(pdf_files, 'rb')
                read_pdf = PdfFileReader(f)
            number_of_pages = read_pdf.getNumPages()
            for i in range(number_of_pages):
                page = read_pdf.getPage(i)
                page_content = page.extractText()
                text+=page_content
        processed_txt = cleantext(text)
        text = ''.join([i for i in processed_txt if not i.isdigit()])
        text = ' '.join(text.split())
    if method == "none":
        logging.info("PDF_NOT_PERFORMED")
    
    logging.info("PTT_STOP")
    return text


def ecml_index_to_text(method, path_to_id):
    all_text = ""
    logging.info("JTT_START")
    if method == "parse":
        if os.path.exists(os.path.join(path_to_id,"index.ecml")) or os.path.exists(os.path.join(path_to_id, "manifest.json")):
            try:
                xmldoc = ET.parse(os.path.join(path_to_id, "index.ecml"))
                xmldoc = minidom.parse(ecml_file)
                logging.info('...File type detected as ecml')
                count_text = 0
                if any(i.tag in ['text', 'htext'] for i in xmldoc.iter()):
                    for elem in xmldoc.iter():
                        if ((elem.tag == 'text') or (elem.tag == 'htext')):
                            count_text += 1
                            try:
                                text = str(elem.text).strip() + ' '
                            except:
                                text = elem.text
                                all_text+=text
                else: 
                    logging.info('........ No text or htext detected')
                    all_text = ""

            except:
                # get json text and translate
                logging.info('...File type detected as json')
                json_data = open(os.path.join(path_to_id, "manifest.json"))
                jdata = json.load(json_data)
                jtext = getText_json(jdata, '__text')

                for text in jtext:  # +jfeedback
                    try:
                        all_text += " " + text
                    except:
                        print('........ Unable to extract json text')
                        all_text = ""
        else:
            logging.info("No json/ecml file detected")
    if method == "none":
        logging.info("JTT_NOT_PERFORMED")
    logging.info("JTT_STOP")
    return all_text


def multimodal_text_enrichment(index, content_meta, content_type, content_to_text_path):
        """
        index, content_meta, content_type, content_to_text_path
        A custom function to extract text from a given Content id in a Content meta dataframe extracted using Content V2 api

        Parameters
        ----------
        index: int
        row id for the Content 

        content_meta: datarame 
        A dataframe of Content metadata.
        Mandatory fields: ['artifactUrl', 'content_type','downloadUrl', 'gradeLevel', 'identifier','keywords', 'language', 'subject'

        content_type: str
        Can be youtube, pdf, ecml, unknown

        content_to_text_path:str
        path to save the extracted text

        Returns
        -------
        path_to_transcript: str
        Path where text is saved
        """
        type_of_url = content_meta.iloc[index]["content_type"]
        id_name = content_meta["identifier"][index]
        downloadField = content_type[type_of_url]["contentDownloadField"]
        url = content_meta[downloadField][index]
        logging.info("MTT_START_FOR_URL {0}".format(url))
        # start text extraction pipeline:
        try:
            path_to_id = download_to_local(type_of_url, url, content_to_text_path, id_name)
            path_to_assets = os.path.join(path_to_id, "assets")
            path_to_audio = video_to_speech(content_type[type_of_url]["video_to_speech"],  path_to_assets)
            textExtraction_pipeline = [(speech_to_text, (content_type[type_of_url]["speech_to_text"], path_to_assets)), 
                                       (image_to_text, (content_type[type_of_url]["image_to_text"],  path_to_assets)),
                                       (pdf_to_text, (content_type[type_of_url]["pdf_to_text"],  path_to_assets, url)),
                                       (ecml_index_to_text, (content_type[type_of_url]["ecml_index_to_text"], path_to_id))]
            path_to_transcript = os.path.join(path_to_id, "enriched_text.txt")
            text = ""
            for method, param_tuple in textExtraction_pipeline:
                text+=method(*param_tuple)
            if os.path.exists(path_to_id):
                with open(path_to_transcript, "w") as myTextFile:
                    myTextFile.write(text)
                logging.info("MTT_TRANSCRIPT_PATH_CREATED: {0}".format(path_to_transcript))
                logging.info("MTT_CONTENT_ID_READ: {0}".format(id_name))
            logging.info("MTT_STOP_FOR_URL {0}".format(url))
            return path_to_transcript
        except:
            logging.info("TextEnrichment failed for url:{0} with id:{1}".format(url, id_name))



def custom_tokenizer(path_to_text_file, path_to_text_tokens_folder):
    """
    Given a text file uses custom_tokenizer function to tokenise and write the tokenised words to a keywords.csv file.

    Parameters
    ----------
    path_to_text_file: str
    Location of text file to be tokenised

    path_to_text_tokens_folder: str
    Location to write the tokenised words

    Returns
    -------
    path: str
    location of keywords file. path_to_pafy_text_tokens_folder+"keywords.csv"
    """
    split_path = os.path.split(path_to_text_file)
    text = open(path_to_text_file, "r")
    text_file = text.read()
    text_list = clean_text_tokens(text_file)
    if not os.path.exists(path_to_text_tokens_folder):
      os.mkdir(path_to_text_tokens_folder)
      os.chdir(path_to_text_tokens_folder)
    else:
      os.chdir(path_to_text_tokens_folder)
    path = os.path.join(path_to_text_tokens_folder, "keywords.csv")
    with open(path, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(['KEYWORDS'])
        for i in text_list:
            wr.writerows([[i]])
    return path


def tagme_text(text):
    url = "https://tagme.d4science.org/tagme/tag"

    querystring = {"lang": "en", "gcube-token": "1e1f2881-62ec-4b3e-9036-9efe89347991-843339462", "text": text}

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


def text_token_taxonomy_intersection_keywords(taxonomy_keywords_set, path_to_text_token, path_to_text_taxonomy_folder):
    try:
        text_token = pd.read_csv(path_to_text_token, sep=',',index_col=None)
        token = [i.lower() for i in list(text_token['KEYWORDS'])]
        common_words = list(set(taxonomy_keywords_set)&set(token))
        if not os.path.exists(path_to_text_taxonomy_folder):
            os.makedirs(path_to_text_taxonomy_folder)

        path = os.path.join(path_to_text_taxonomy_folder, 'keywords.csv')
        with open(path, 'w') as myfile:
          wr = csv.writer(myfile)
          wr.writerow(['KEYWORDS'])
          for i in common_words:
              wr.writerows([[i]])
        return path
    except:
        logging.info("Keywords cannot be extracted")


def tagme_taxonomy_intersection_keywords(taxonomy_keywords_set, path_to_tagme_keys, path_to_tagme_taxonomy_folder):
    try:
        split_path = os.path.split(path_to_tagme_keys)
        tagme_spots = pd.read_csv(path_to_tagme_keys, sep=',', index_col=None)
        spots = [i.lower() for i in list(tagme_spots['KEYWORDS'])] 
        common_words = list(set(taxonomy_keywords_set) & set(spots))
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


def keyword_extraction_parallel(dir, content_to_text_path, taxonomy, extract_keywords, filter_criteria):
        """
        A custom function to parallely extract keywords for all the Content texts in a folder. This is run typically after multimodal_text_enrichment.
        Part of Content enrichment pipeline.
        The funtion allows keyword extraction using TAGME or tokenising the words using nltk tokeniser.
        The extracted keywords can be filtered based on following criteris:
            -taxonomy: if the word is a taxonomy keyword
            -dbpedia: if the keyword is domain keyword based on dbpedia criteria(occurs under the domain ontolgy in wikipedia)
            - none


        Parameters
        ----------
        dir: str
        Name of the folder containing enriched_text.txt file inside it.

        content_to_text_path:str
        path to directory containing multiple Content id folders

        taxonomy: str
        path to taxonomy file(csv)

        extract_keywords:str
        can be "tagme" or "text_token"

        filter_criteria: str
        can be "taxonomy" or "none"


        Returns
        -------
        content_to_text_path: str
        path to extracted keywords.
        """
        print("*******dir*********:", dir)
        print("***Extract keywords***:", extract_keywords)
        print("***Filter criteria:***", filter_criteria)
        path_to_cid_transcript = os.path.join(content_to_text_path, dir, "enriched_text.txt")
        keywords = os.path.join(content_to_text_path, dir, "keywords")
        path_to_keywords = os.path.join(keywords, extract_keywords+"_"+filter_criteria)
        if os.path.isfile(path_to_cid_transcript):
            logging.info("Transcript present for cid: {0}".format(dir))
#             try:
            if os.path.getsize(path_to_cid_transcript) > 0:
                print("Path to transcripts ", path_to_cid_transcript)
                print("Running keyword extraction for {0}".format(path_to_cid_transcript))
                print("---------------------------------------------")

                if extract_keywords == "tagme" and filter_criteria == "none":
                    print("Tagme keyword extraction is running for {0}".format(path_to_cid_transcript))
                    path_to_tagme_keywords = get_tagme_longtext(path_to_cid_transcript, path_to_keywords)
                    logging.info("Path to tagme tokens is {0}".format(path_to_tagme_keywords))

                elif extract_keywords == "text_token" and filter_criteria == "none":
                    print("Text tokens extraction running for {0}".format(path_to_cid_transcript))
                    path_to_text_tokens = custom_tokenizer(path_to_cid_transcript, path_to_keywords)
                    print("Path to text tokens is {0}".format(path_to_text_tokens))

                elif extract_keywords == "text_token" and filter_criteria == "taxonomy":
                    print("Text tokens intersection taxonomy running for {0}".format(path_to_cid_transcript))
                    if not os.path.exists(os.path.join(keywords, "text_token_none")):
                        path_to_text_tokens = custom_tokenizer(path_to_cid_transcript, os.path.join(keywords, "text_token"))
                    else:
                        path_to_text_tokens = os.path.join(keywords, "text_token")
                    clean_keywords = map(get_words, list(taxonomy["Keywords"]))
                    clean_keywords = map(clean_string_list, clean_keywords)
                    flat_list = [item for sublist in list(clean_keywords) for item in
                                 sublist]
                    taxonomy_keywords_set = set([cleantext(i) for i in flat_list])
                    path_to_text_intersect_tax = text_token_taxonomy_intersection_keywords(taxonomy_keywords_set, path_to_text_tokens, path_to_keywords)
                    print \
                        ("Path to text tokens intersection taxonomy is {0}".format(path_to_text_intersect_tax))

                elif extract_keywords == "tagme" and filter_criteria == "taxonomy":
                    print("Tagme intersection taxonomy keyword extraction is running for {0}".format(
                        path_to_cid_transcript))
                    clean_keywords = map(get_words, list(taxonomy["Keywords"]))
                    clean_keywords = map(clean_string_list, clean_keywords)
                    flat_list = [item for sublist in list(clean_keywords) for item in
                                 sublist]
                    taxonomy_keywords_set = set([cleantext(i) for i in flat_list])
                    path_to_tagme_keywords = get_tagme_longtext(path_to_cid_transcript, os.path.join(keywords, "tagme"))
                    path_to_tagme_intersect_tax = tagme_taxonomy_intersection_keywords(taxonomy_keywords_set,
                                                                                       path_to_tagme_keywords,
                                                                                       path_to_keywords)
                    print \
                        ("Path to tagme taxonomy intersection tokens is {0}".format(path_to_tagme_intersect_tax))

                else:
                    print("Invalid argument provided")


            else:
                print("The text file {0} has no contents".format(path_to_cid_transcript))

#             except:
#                 print("Raise exception for {0} ".format(path_to_cid_transcript))
#                 logging.info("Raise exception for {0} ".format(path_to_cid_transcript))
        else:
            print("Transcripts doesnt exist for {0}".format(path_to_cid_transcript))

        return content_to_text_path

def clean_string_list(x_list):
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
        x=[strip_word(i, DELIMITTER) for i in x]
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


def get_level_keywords(taxonomy_df, level):
    level_keyword_df=[]
    for subject in list(set(taxonomy_df[level])):
        Domain_keywords= list(taxonomy_df.loc[taxonomy_df[level]==subject,'Keywords'])
        unique_keywords=[ind for sublist in Domain_keywords for ind in sublist]
        level_keyword_df.append({level:subject,'Keywords':unique_keywords})
    level_keyword_df=pd.DataFrame(level_keyword_df)
    return level_keyword_df


def getGradedigits(class_x):
    for i in ["Class", "[", "]", " ", "class", "Grade", "grade"]:
        class_x=class_x.replace(i, "")
    return class_x


def match_str_list(list1, list2):
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


def jaccard_with_phrase(list1, list2):
    DELIMITTER = "_"
    intersection_words = []
    list1 = list(set(list1))
    list2 = list(set(list2))
    list1 = removeShortWords(list1, 0)
    list2 = removeShortWords(list2, 0)
    list1_words, list1_phrases=getPhrase(list1, DELIMITTER)
    list2_words, list2_phrases=getPhrase(list2, DELIMITTER)
    list1_len = len(list1_words)+len(list1_phrases)
    list2_len = len(list2_words)+len(list2_phrases)
    intersection = 0
    match_count = 0
    # count matching words
    exact_word_intersection = list(set(list1_words) & set(list2_words))
    intersection_words.extend([(a, a) for a in exact_word_intersection])
    exact_word_match = match_str_list(list1_words,list2_words)['intersection']
    intersection = intersection+exact_word_match
    match_count += exact_word_match
    phraselist1=list1_phrases
    phraselist2=list2_phrases
    exact_phrase_intersection=[]
    for phrase1 in phraselist1:
        for phrase2 in phraselist2:
            if((phrase2 in phrase1)  or (phrase1 in phrase2)):
                exact_phrase_intersection.append((phrase1,phrase2) )
                list2_phrases.remove(phrase2)
                break
    exact_phrase_length=sum([min([(len(j.split(DELIMITTER))) for j in i ]) for i in exact_phrase_intersection])
    intersection+=(2.0*exact_phrase_length)
    match_count+=len(exact_phrase_intersection)
    intersection_words.extend(exact_phrase_intersection)  
    non_matched_list1_words,non_matched_list2_words=list1_words, list2_words
    non_matched_list1_phrases,non_matched_list2_phrases=list1_phrases,list2_phrases
    if exact_word_intersection:
        non_matched_list1_words=[ item for item in list1_words if str(item) not in exact_word_intersection ]
        non_matched_list2_words=[ item for item in list2_words if str(item) not in exact_word_intersection ]
    if exact_phrase_intersection:
        non_matched_list1_phrases=[word for item in exact_phrase_intersection for word in non_matched_list1_phrases if item[0] not in word]   
        non_matched_list2_phrases=[word for item in exact_phrase_intersection for word in non_matched_list2_phrases if item[1] not in word]   
    partial_match_list1,count=WordtoPhraseMatch(non_matched_list1_words,non_matched_list2_phrases,DELIMITTER)
    match_count+=count
    if partial_match_list1:
        non_matched_list1_words=[word for item in partial_match_list1 for word in non_matched_list1_words if item[0] not in word]
        non_matched_list2_phrases=[word for item in partial_match_list1 for word in non_matched_list2_phrases if item[1] not in word]   
    intersection=intersection+len(partial_match_list1)
    intersection_words.extend(partial_match_list1)
    # Content phrase to taxonomy words
    partial_match_list2,count=WordtoPhraseMatch(non_matched_list2_words,non_matched_list1_phrases,DELIMITTER)
    match_count+=count
    non_matched_list2_words=[item[0] for item in partial_match_list2 if item[0] not in non_matched_list1_phrases]
    intersection=intersection+len(partial_match_list2)
    intersection_words.extend(partial_match_list2)
    intersection_words=[el for el in intersection_words if el!=[]]

    if (((len(list2))!=0) & ((len(list1)+len(list2)-match_count)!=0)):
        return {'jaccard':float(intersection)/float(len(list1)+len(list2)-match_count),'match_percentage':float(intersection)/float(len(list2)),'word_intersection':intersection_words}
    elif ((len(list1)+len(list2)-match_count)==0):

        return {'jaccard':0,'match_percentage':float(intersection)/float(len(list2)),'word_intersection':intersection_words}
    elif (len(list2)==0):
        return {'jaccard':float(intersection)/float(len(list1)+len(list2)-match_count),'match_percentage':0,'word_intersection':intersection_words}


def precision_from_dictionary(predicted_df, observed_df, window_len):
    window = range(1,window_len+1)
    percent_list = []
    for ind in window:
        count = 0
        for cid in predicted_df.index:
            try:
                if (observed_df.loc[cid].values[0]) in list(predicted_df.loc[cid][0:ind]):
                    count += 1
            except:
                print(str(cid)+" metadata not available")
        percent_list.append(count*100.0/len(predicted_df.index))
    return pd.DataFrame(percent_list, index=window, columns=["percent"])


def agg_precision_from_dictionary(predicted_dct, observed_dct, window_len):
    predicted_val_list = []
    observed_val_list = []
    for subject in observed_dct.keys():
        predicted_df = predicted_dct[subject]
        for i in range(predicted_df.shape[0]):
            predicted_val_list.append(list(predicted_df.iloc[i]))
            observed_val_list.append(observed_dct[subject].iloc[i][0])
    window = [0]*window_len
    for i in range(0, window_len):
        count = 0
        for ind in range(len(predicted_val_list)):
            if observed_val_list[ind] in predicted_val_list[ind][0:min(len(predicted_val_list[ind]), i+1)]:
                count += 1
        window[i] = 100.0*count/len(predicted_val_list)
    return pd.DataFrame(window, columns=['Percent'])
