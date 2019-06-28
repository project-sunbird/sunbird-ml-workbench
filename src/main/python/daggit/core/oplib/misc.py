import yaml
import glob
import numpy as np
import pandas as pd
import pickle
import urllib.parse
import urllib
import shutil
import string
import zipfile
import nltk
from nltk.corpus import stopwords

import json
import re
import io
import time
import requests
import os
import logging
from dateutil.parser import parse
from SPARQLWrapper import SPARQLWrapper, JSON
logging.getLogger("requests").setLevel(logging.WARNING)


from google.cloud.vision import types
from google.cloud import vision
from google.cloud import translate


stopwords = stopwords.words('english')
pun_list = list(string.punctuation)


def is_date(date_string):

    """
    This function  takes a string as an argument and check if the string is a valid date
    :param date_string(str): A string
    :returns: A boolean value. True if the string is a valid date and False if otherwise
    """
    try:
        parse(date_string)
        return True
    except ValueError:
        return False


def df_feature_check(df, mandatory_fields):
    """
    Check if columns are present in the dataframe.
    :param df(dataframe): DataFrame that needs to be checked
    :param mandatory_fields(list of strings): List of column names.``eg: jpg, png, webm, mp4``
    :returns: ``True`` if all columns are present and ``False`` if not all columns are present
    """
    check = [0 if elem in list(df.columns) else 1 for elem in mandatory_fields]
    if sum(check) > 0:
        return False
    else:
        return True


def identify_contentType(url):

    """
    Given a URL for a content, it identifies the type of the content
    :param url(str): URL
    :returns: Type of the content
    """
    extensions = ['mp3', 'wav', 'jpeg', 'zip', 'jpg', 'mp4', 'webm', 'ecar', 'wav', 'png']
    if ('youtu.be' in url) or ('youtube' in url):
        return "youtube"
    elif url.endswith('pdf'):
        return "pdf"
    elif any(url.endswith(x) for x in extensions):
        return "ecml"
    else:
        return "unknown"


def fetch_video_id(url):

    """
    Parse a youtube URL and generate video id
    :param url(str): youtube video URL
    :returns: Video id of the video URL
    """
    parse_url = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parse_url.query)
    return query["v"][0]


def translate_target_language(text, target_lan):
    """Translates a given text into target language
    :param text(str): Text that need to be translated
    :returns: Text translated into a target language, say:``en(English)``
    """
    translate_client = translate.Client()
    translation = translate_client.translate(
        text, target_language=target_lan)
    return translation['translatedText']


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def dictionary_merge(dict_ls):

    keys_ls = [dictionary.keys() for dictionary in dict_ls]
    keys_ls = set([elem for sublist in keys_ls for elem in sublist])
    dict_all = {keys: pd.DataFrame() for keys in (keys_ls)}
    for dictionary in dict_ls:
        for keys in dictionary.keys():
            dict_all[keys] = dict_all[keys].append(dictionary[keys])
    return dict_all


def get_sorted_list(x, order):
    x_df = pd.DataFrame(x)
    return list(x_df.sort_values(by=list(x_df.columns), ascending=order).index)


def CustomDateFormater(**kwargs):
    import datetime
    from datetime import date, timedelta
    expected_args = ['x', 'fileloc', 'datepattern']
    kwargsdict = dict()
    for key in kwargs.keys():
        if key in expected_args:
            kwargsdict[key] = kwargs[key]
        else:
            raise Exception("Unexpected Argument")
    if kwargsdict['x']:
        x = kwargsdict['x']
        if x == "today":
            dateobj = date.today()
        elif x == "yesterday":
            dateobj = date.today()-timedelta(1)
        elif x == "lastrun":
            list_of_files = glob.glob(kwargsdict['fileloc']+"/*")
            timestr_files = [file for file in list_of_files if is_date(os.path.split(file)[1])]
            latest_file = max(timestr_files, key=os.path.getctime).split("/")[-1]
            dateobj = datetime.datetime.strptime(latest_file, kwargsdict['datepattern'])
        elif isinstance(x, str):
            dateobj = datetime.datetime.strptime(x, kwargsdict['datepattern'])
        elif isinstance(x, datetime.date):
            dateobj = x
        return dateobj.strftime('%Y-%m-%dT%H:%M:%S.000+0530')
    else:
        raise Exception("Require atleast 1 argument.")


def findDate(x_date, DS_DATA_HOME):
    if x_date in ['today', 'yesterday']:
        return CustomDateFormater(x=x_date)
    elif x_date == "lastrun":
        return CustomDateFormater(x=x_date, fileloc=DS_DATA_HOME, datepattern='%Y%m%d-%H%M%S')
    else:
        return CustomDateFormater(x=x_date, datepattern="%d-%m-%Y")


def merge_json(dict1, dict2, path=None):
   if path is None: path = []
   for key in dict2:
       if key in dict1:
           if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
               merge_json(dict1[key], dict2[key], path + [str(key)])
           elif dict1[key] == dict2[key]:
               pass # same leaf value
           else:
               raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
       else:
           dict1[key] = dict2[key]
   return dict1


def embed_youtube_url_validation(url):
    """
    Convert a broken youtube URL to custom youtube URL
    :param url(str): Youtube URL
    :returns: Custom youtube URL
    """
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

    youtube_regex_match = re.match(youtube_regex, url)
    if youtube_regex_match:
        if youtube_regex_match.group(6) == "https://www":
            return "https://www.youtube.com/watch?v=" + \
                fetch_video_id(url.split("embed/")[1])[:11]
        else:
            return "https://www.youtube.com/watch?v=" + \
                youtube_regex_match.group(6)

