import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from natsort import natsorted
from pydub import AudioSegment
from google.cloud.vision import types
from google.cloud import vision
from google.cloud import translate
from nltk.corpus import stopwords
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import HTMLConverter, TextConverter, XMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from kafka import KafkaProducer, KafkaConsumer, KafkaClient

import yaml
import glob
from PyPDF2 import PdfFileReader
import numpy as np
import pandas as pd
import ffmpy
import speech_recognition as sr
import youtube_dl
import pickle
import urllib.parse
import urllib
import shutil
import string
import zipfile
import json
import re
import io
import time
import requests
import os
import logging
import redis
import xml.etree.ElementTree as ET
from dateutil.parser import parse
from SPARQLWrapper import SPARQLWrapper, JSON
logging.getLogger("requests").setLevel(logging.WARNING)

# nltk.download("stopwords")
# nltk.download("wordnet")
stopwords = stopwords.words('english')


pun_list = list(string.punctuation)

from daggit.core.oplib.misc import embed_youtube_url_validation
from daggit.core.oplib.nlp import clean_text, language_detection
from daggit.core.oplib.nlp import clean_string_list
from daggit.core.oplib.nlp import get_tokens
from daggit.core.io.redis import setRediskey, getRediskey
from daggit.core.io.files import findFiles
from daggit.core.io.files import downloadZipFile
from daggit.core.io.files import unzip_files

def download_file_to_folder(url_to_download, path_to_folder, file_name): #download_from_downloadUrl
    download_dir = os.path.join(path_to_folder, 'temp' + file_name)
    status = downloadZipFile(url_to_download, download_dir)
    try:
        if status:
            unzip_files(download_dir)
            ecar_unzip(
                download_dir, os.path.join(
                    path_to_folder, file_name))
            path_to_file = os.path.join(path_to_folder, file_name)
            return path_to_file
    except BaseException:
        print("Unavailable for download")



def ecar_unzip(download_location, copy_location): #ekstep_ecar_unzip
    """
    This function unzips an ecar file(ekstep file format)
    and parses all the subfolder.
    All the files are copied into one of ``'assets','data','items'`` folder
    (same name as in downloaded folder is maintained)
    based on its location in the downloaded folder.
    :param download_location(str): A location in the disk where ekstep ecar resource file is downloaded
    :param copy_location(str): A disk location where the ecar is unwrapped
    """
    assert isinstance(download_location, str)
    assert isinstance(copy_location, str)
    if not os.path.exists(copy_location):
        os.makedirs(copy_location)
    #To make the new sub-directories in which the files will be eventually stored
    location=[os.path.join(copy_location,folder) for folder in ['assets','data','items']]
    for loc in location:
        if not os.path.exists(loc):
            os.makedirs(loc)
    ecar_extensions = ['png', 'gif', 'jpg', 'mp4', 'webm', 'pdf', 'mp3', 'ecml']
    files_found = findFiles(download_location, ecar_extensions)
    if files_found:
        for file in files_found:
            if file[-4:] in "ecml":
                shutil.copy(file, copy_location)
            else:
                shutil.copy(file, os.path.join(copy_location, "assets"))
    else:
        print("No files to copy!")
    # Delete the messy download directory
    if os.path.exists(download_location):
        shutil.rmtree(download_location)


def getImgTags(img_file_name):
    """
    Enables to detect text from an image file using Google cloud vision API
    :param img_file_name(str): Path to an image file
    :returns: Text detected from the image file
    """
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
    """
    Download audio in .mp3 format from a youtube URL and save it in a disk location.
    :param url(str): A youtube URL
    :returns: Path to the downloaded audio
    """
    logging.info("UTAE_YOUTUBE_URL_START: {0}".format(url))
    if not os.path.exists(path):
        os.makedirs(path)
    cid = os.path.split(os.path.split(path)[0])[1]
    path_to_audio = os.path.join(path, cid + ".mp3")
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
        os.rename(
            list(
                filter(
                    (lambda x: '.mp3' in x),
                    os.listdir(path)))[0],
            path_to_audio)
        logging.info("UTAE_AUDIO_DOWNLOAD_COMPLETE")
        return path_to_audio
    else:
        return path_to_audio
    logging.info("UTAE_YOUTUBE_URL_STOP")


def audio_split(path_to_audiofile, path_to_split_audio):
    """
    Takes in an audiofile, split it as per the duration and
    returns the path to the split
    :param path_to_audiofile(str): The string is the path to the audio m4a file.
    :param path_to_split_audio(str): The path to save the audio segmented according to the duration.
    :returns: Path to audio segment for a given audio file.
    """
    if not os.path.exists(path_to_split_audio):
        os.makedirs(path_to_split_audio)
        split_path = os.path.split(path_to_audiofile)
        sample_audio = AudioSegment.from_mp3(path_to_audiofile)
        if sample_audio.frame_rate > 16000:
            sample_audio = sample_audio.set_frame_rate(16000)
        limit_sec = 59 * 1000
        no_of_splits = int(np.ceil(sample_audio.duration_seconds / 59))
        for split in range(no_of_splits):
            batch = sample_audio[limit_sec *
                                 (split):min(limit_sec *
                                             (split +
                                              1), len(sample_audio))]
            path = os.path.join(path_to_split_audio, split_path[1][:-4])
            if not os.path.exists(path):
                os.mkdir(path)
                batch.export(path +
                             "/" +
                             split_path[1][:-
                                           4] +
                             "_" +
                             str(split) +
                             ".mp3", format="mp3")
            else:
                batch.export(path +
                             "/" +
                             split_path[1][:-
                                           4] +
                             "_" +
                             str(split) +
                             ".mp3", format="mp3")
        return path
    else:
        r = re.search('(.*)/assets', path_to_split_audio)
        id = os.path.split(r.group(1))[1]
        return os.path.join(path_to_split_audio, id)


def getTexts(AUDIO_FILE, lan, GOOGLE_APPLICATION_CREDENTIALS):
    """
    Takes in an audiofile and return text of the given audio input
    :param AUDIO_FILE(str):     path to audio file in mp3 format
    :param lan(str): This attribute will set the language of the recognition
    :returns: Recognized text for the audio snippet
    """
    temp = AUDIO_FILE[:-4]
    AudioSegment.from_file(
        '%s.mp3' %
        (temp)).export(
        '%s.wav' %
        (temp), format='wav')
    AUDIO_FILE = '%s.wav' % (temp)
    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file

    try:
        audio_text = r.recognize_google_cloud(
            audio,
            credentials_json=GOOGLE_APPLICATION_CREDENTIALS,
            language=lan)
    except sr.UnknownValueError:
        audio_text = ''
    except sr.RequestError:
        audio_text = ''
    mp3_dct = {"text": audio_text}
    return mp3_dct


def audio_to_text(path_to_audio_split_folder, GOOGLE_APPLICATION_CREDENTIALS):

    """
    This function takes in a folder of audios split as per its time duration and convert each audio into text
    in succession and returns the concatenated texts for a given audio split folder
    :param path_to_audio_split_folder(str): Path to a folder with the audio splits
    :param GOOGLE_APPLICATION_CREDENTIALS: Environment variable to designate path to CREDENTIALS json
    :returns: Concatenated text
    """
    text = ""
    for i in natsorted(os.listdir(path_to_audio_split_folder), reverse=False):
        if i[-4:] == ".mp3":
            try:
                text += getTexts(os.path.join(path_to_audio_split_folder,
                                              i), 'en-IN', GOOGLE_APPLICATION_CREDENTIALS)['text']
            except LookupError:
                continue
    return text


def getText_json(jdata, key):
    sdata = json.dumps(jdata)
    text_list = ([sdata[(m.start(0) + len(key) + 4):m.end(0) - 1]
                  for m in re.finditer(key + '": "(.*?)"', sdata)])
    return text_list


def download_content(method, url_to_download, path_to_save, id_name): #download_to_local
    logging.info("DTL_START_FOR_URL: {0}".format(url_to_download))
    path_to_id = ""
    if method == "ecml":
        logging.info("DTL_ECAR_URL: {0}".format(url_to_download))
        try:
            path_to_id = download_file_to_folder(
                url_to_download, path_to_save, id_name)
        except RuntimeError:
            logging.info("Skipped url: {0}".format(url_to_download))

    if method == "youtube":
        try:
            logging.info("DTL_YOUTUBE_URL: {0}".format(url_to_download))
            path_to_id = os.path.join(path_to_save, id_name)
            location = [os.path.join(path_to_id, folder)
                        for folder in ['assets', 'data', 'items']]

            path_to_audio_download = os.path.join(path_to_id, "assets")
            for loc in location:
                if not os.path.exists(loc):
                    os.makedirs(loc)
            path_to_audio = url_to_audio_extraction(
                url_to_download, path_to_audio_download)
            logging.info("Path to audio file is {0}".format(path_to_audio))
        except BaseException:
            logging.info("Could not download the youtube url")

    if method == "pdf":
        logging.info("DTL_PDF_URL: {0}".format(url_to_download))
        try:
            path_to_id = download_file_to_folder(
                url_to_download, path_to_save, id_name)
        except BaseException:
            logging.info("Skipped url: {0}".format(url_to_download))

    else:
        logging.info(
            "Download not required for url: {0}".format(url_to_download))
    logging.info("DTL_STOP_FOR_URL: {0}".format(url_to_download))
    return path_to_id


def video_to_speech(method, path_to_assets):
    logging.info('VTS_START')
    video_names = findFiles(path_to_assets, ['mp4', 'webm'])
    logging.info('...detected {0} video files'.format(str(len(video_names))))
    if method == "ffmpeg" and len(video_names) > 0:
        logging.info("VTS_START_FOR_METHOD: {0}".format(method))

        for file in video_names:
            try:
                # ffmpy wrapper to convert mp4 to mp3:
                if "webm" in file:
                    ff = ffmpy.FFmpeg(inputs={file: None}, outputs={os.path.join(
                        file[:-5] + ".mp3"): '-vn -ar 44100 -ac 2 -ab 192 -f mp3'})
                else:    
                    ff = ffmpy.FFmpeg(inputs={file: None}, outputs={os.path.join(
                        file[:-4] + ".mp3"): '-vn -ar 44100 -ac 2 -ab 192 -f mp3'})
                ff.run()
                if os.path.exists(os.path.join(
                        path_to_assets, file[:-4] + ".mp3")):
                    path_to_audio = os.path.join(
                        path_to_assets, file[:-4] + ".mp3")
                    logging.info("VTS_AUDIO_DOWNLOAD_PATH: ".format(path_to_audio))
                else:
                    logging.info("mp3 download unsuccessful")
            except:
                logging.info("Unable to convert audio file: {0}".format(file))
    if method == "none":
        logging.info("No Video content detected")
    logging.info('VTS_STOP')
    return path_to_assets


def speech_to_text(method, path_to_assets, GOOGLE_APPLICATION_CREDENTIALS):
    logging.info("STT_START")
    text = ""
    if not os.path.exists(path_to_assets):
        logging.info("No audio file detected")
    else:
        audio_names = findFiles(path_to_assets, ['mp3'])
        if method == "googleAT" and len(audio_names) > 0:
            try:
                for i in audio_names:
                    logging.info(
                        "STT_AUDIO_FILEPATH: {0}".format(
                            os.path.join(
                                path_to_assets, i)))
                    path_to_split = audio_split(
                        os.path.join(
                            path_to_assets, i), os.path.join(
                            path_to_assets, "audio_split"))
                    logging.info("STT_AUDIO_SPLIT: {0}".format(path_to_split))
                    text += audio_to_text(path_to_split, GOOGLE_APPLICATION_CREDENTIALS)
            except BaseException:
                text = ""
        elif method == "none":
            logging.info("STT_NOT_PERFORMED")
        else:
            logging.info("Unknown method given")
    logging.info("STT_STOP")
    text_dict = {"text": text}
    return text_dict


def image_to_text(method, path_to_assets):
    logging.info("ITT_START")
    image_text = ""
    image_names = findFiles(path_to_assets, ['png', 'gif', 'jpg'])
    if method == "googleVision" and len(image_names) > 0:
        try:
            logging.info('...detected {0} video files'.format(
                str(len(image_names))))
            logging.info('...image file processing started')
            for file in image_names:
                try:
                    image_text += getImgTags(file)
                except BaseException:
                    print('........ Error: could not process file')
            print("Text: ", image_text)
            text = list(str(image_text.lower()).split("\n"))
            image_text = ' '.join(list(set(text)))
        except BaseException:
            image_text = ""
    if method == "none":
        logging.info("ITT_NOT_PERFORMED")
    logging.info("ITT_STOP")
    text_dict = {"text": image_text}
    return text_dict


def convert_pdf_to_txt(path_to_pdf_file):
    """
    A basic wrapper around PDF miner that parse a PDF file and extracts text from it.
    :param path_to_pdf_file(str): Path to a PDF file
    :returns: Text extracted from the PDF file
    """
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path_to_pdf_file, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
        interpreter.process_page(page)
    text = retstr.getvalue()
    fp.close()
    device.close()
    retstr.close()
    return text


def pdf_to_text(method, path_to_assets, pdf_url):
    """
    """
    text = ""
    number_of_pages = 0
    logging.info("PTT_START")
    pdf_names = findFiles(path_to_assets, ['.pdf'])
    print("----->pdf_names: ", pdf_names)
    if method == "PyPDF2":
        logging.info("PTT_METHOD: {0}".format(method))
        for j in range(0, len(pdf_names) + 1):
            if (len(pdf_names) == 0 and pdf_url.endswith('pdf')):
                r = requests.get(pdf_url)
                f = io.BytesIO(r.content)
                read_pdf = PdfFileReader(f)
                number_of_pages = read_pdf.getNumPages()
            elif j < (len(pdf_names)):
                pdf_files = pdf_names[j]
                f = open(pdf_files, 'rb')
                read_pdf = PdfFileReader(f)
                number_of_pages = read_pdf.getNumPages()
            else:
                number_of_pages = 0
    if method == "pdfminer":
        logging.info("PTT_METHOD: {0}".format(method))
        text = ""
        for j in range(0, len(pdf_names) + 1):
            if (len(pdf_names) == 0 and pdf_url.endswith('pdf')):
                r = requests.get(pdf_url)
                f = io.BytesIO(r.content)
                read_pdf = PdfFileReader(f)
                number_of_pages = read_pdf.getNumPages()
            elif j < (len(pdf_names)):
                pdf_files = pdf_names[j]
                text = ""
                text = convert_pdf_to_txt(pdf_files)
                number_of_pages = 0
            else:
                number_of_pages = 0
    if method == "none":
        logging.info("PDF_NOT_PERFORMED")
    if number_of_pages > 0:
        for i in range(number_of_pages):
            page = read_pdf.getPage(i)
            page_content = page.extractText()
            text += page_content
    processed_txt = clean_text(text)
    text = ''.join([i for i in processed_txt if not i.isdigit()])
    text = ' '.join(text.split())
    logging.info("PTT_STOP")
    text_dict = {"text": text, "no_of_pages": number_of_pages}
    return text_dict


def ecml_parser(ecml_file):
    all_text = ""
    xmldoc = ET.parse(ecml_file)
    if any(i.tag in ['text', 'htext'] for i in xmldoc.iter()):
        for elem in xmldoc.iter():
            if ((elem.tag == 'text') or (elem.tag == 'htext')):
                try:
                    text = str(elem.text).strip() + ' '
                except BaseException:
                    text = elem.text
                    all_text += text
    else:
        all_text = ""
    plugin_used = []
    for elem in xmldoc.iter():
        if (elem.tag in ['media']):
            plugin_used.append(elem.attrib['plugin'])
        if (elem.tag in ['plugin']):
            plugin_used.append(elem.attrib['id'])
    plugin_used = list(set(plugin_used))
    num_stages = [i.tag for i in xmldoc.iter()].count('stage')
    return dict({'text': all_text, 'plugin': plugin_used, 'stages': num_stages})


def modified_ecml_parser(ecml_file):
    tree = ET.parse(ecml_file)
    root = tree.getroot()
    tag_to_check = ['org.ekstep.text']
    alltext = ""
    # create empty list for text items 
    for i in root.iter():
        if i.tag in tag_to_check: 
            content = ""
            for item in root.findall('./stage/'+i.tag): 
                for child in item:
                    try:
                        if child.tag == 'config': 
                            result = re.split(r"\,", child.text)
                            content = str(result[8][7:]).strip() + ' '
                    except:
                        pass
                    alltext+=content
        else:
            continue
    plugin_used = []
    num_stages = []
    try:
        for elem in root.iter():
            if (elem.tag in ['media']):
                plugin_used.append(elem.attrib['plugin'])
            if (elem.tag in ['plugin']):
                plugin_used.append(elem.attrib['id'])
    except:
        pass
    plugin_used = list(set(plugin_used))
    num_stages = [i.tag for i in tree.iter()].count('stage')
    return dict({'text': alltext, 'plugin': plugin_used, 'stages': num_stages})


def ecml_index_to_text(method, path_to_id):
    all_text = ""
    plugin_used = []
    num_stages = 0
    logging.info("ETT_START")
    if method == "parse":
        if os.path.exists(os.path.join(path_to_id, "index.ecml")):
            ecml_file = os.path.join(path_to_id, "index.ecml")
            try:
                logging.info('...File type detected as ecml')
                ecml_tags = modified_ecml_parser(ecml_file)
                all_text = ecml_tags['text']
                plugin_used = ecml_tags['plugin']
                num_stages = ecml_tags['stages']
            except BaseException:
                logging.info("ecml_text cannot be extracted")

        elif os.path.exists(os.path.join(path_to_id, "manifest.json")):
            # get json text and translate
            logging.info('...File type detected as json')
            json_data = open(os.path.join(path_to_id, "manifest.json"))
            jdata = json.load(json_data)
            jtext = getText_json(jdata, '__text')
            if jtext:
                for text in jtext:
                    try:
                        all_text += " " + text
                    except BaseException:
                        print('........ Unable to extract json text')
                        all_text = ""
            else:
                logging.info("jtext empty")

        else:
            logging.info("No json/ecml file detected")
    if method == "none":
        logging.info("ETT_NOT_PERFORMED")
    logging.info("ETT_STOP")
    text_dict = {"text": all_text, "plugin_used": plugin_used, 'num_stage': num_stages}
    return text_dict


def multimodal_text_enrichment(
        index,
        timestr,
        content_meta,
        content_type,
        content_to_text_path,
        GOOGLE_APPLICATION_CREDENTIALS):
    """
    A custom function to extract text from a given
    Content id in a Content meta dataframe extracted using Content V2 api
    Parameters
    ----------
    :param index(int): row id for the Content
    :param content_meta(dataframe): A dataframe of Content metadata.
     Mandatory fields for content_meta are ``['artifactUrl', 'content_type','downloadUrl',
     'gradeLevel', 'identifier','keywords',
     'language', 'subject']``
    :param content_type(str): Can be ``youtube, pdf, ecml, unknown``
    :param content_to_text_path(str): path to save the extracted text
    :returns: Path where text is saved
    """
    type_of_url = content_meta.iloc[index]["derived_contentType"]
    id_name = content_meta["identifier"][index]
    downloadField = content_type[type_of_url]["contentDownloadField"]
    url = content_meta[downloadField][index]
    logging.info("MTT_START_FOR_INDEX {0}".format(index))
    logging.info("MTT_START_FOR_CID {0}".format(id_name))
    logging.info("MTT_START_FOR_URL {0}".format(url))
    # start text extraction pipeline:
    try:
        start = time.time()
        path_to_id = download_content(
            type_of_url, url, content_to_text_path, id_name)
        print("path_to_id", path_to_id)
        path_to_assets = os.path.join(path_to_id, "assets")
        if type_of_url != "pdf":
            path_to_audio = video_to_speech(
                content_type[type_of_url]["video_to_speech"],
                path_to_assets)
            print(path_to_audio)
        if len(findFiles(path_to_assets, ["mp3"])) > 0:
            audio = AudioSegment.from_mp3(findFiles(path_to_assets, ["mp3"])[0])
            duration = round(len(audio) / 1000)
        else:
            duration = 0
        textExtraction_pipeline = [
            (speech_to_text,
             (content_type[type_of_url]["speech_to_text"],
              path_to_assets, GOOGLE_APPLICATION_CREDENTIALS)),
            (image_to_text,
             (content_type[type_of_url]["image_to_text"],
              path_to_assets)),
            (pdf_to_text,
             (content_type[type_of_url]["pdf_to_text"],
              path_to_assets,
              url)),
            (ecml_index_to_text,
             (content_type[type_of_url]["ecml_index_to_text"],
              path_to_id))]
        path_to_transcript = os.path.join(path_to_id, "enriched_text.txt")
        text = ""
        for method, param_tuple in textExtraction_pipeline:
            text += method(*param_tuple)["text"]
        # Adding description and title to the text only for PDF content
        if type_of_url == "pdf":
            text += content_meta["name"].iloc[index] + " " + content_meta["description"].iloc[index]
        if os.path.exists(path_to_id) and text:
            with open(path_to_transcript, "w") as myTextFile:
                myTextFile.write(text)
        # num_of_PDFpages = pdf_to_text("none", path_to_assets, url)["no_of_pages"]
        # Reading pdata
        airflow_home = os.getenv('AIRFLOW_HOME', os.path.expanduser('~/airflow'))
        dag_location = os.path.join(airflow_home, 'dags')
        print("AIRFLOW_HOME: ", dag_location)
        filename = os.path.join(dag_location, 'graph_location')
        f = open(filename, "r")
        lines = f.read().splitlines()
        pdata = lines[-1]
        f.close()

        # estimating ets:
        epoch_time = time.mktime(time.strptime(timestr, "%Y%m%d-%H%M%S"))
        domain = content_meta["subject"][index]
        object_type = content_meta["objectType"][index]
        template = ""
        plugin_used = []
        num_of_stages = 0
        # only for type ecml
        if type_of_url == "ecml":
            plugin_used = ecml_index_to_text("parse", path_to_id)["plugin_used"]
            num_of_stages = ecml_index_to_text("parse", path_to_id)["num_stage"]
        
        mnt_output_dict_new =  { "ets" : int(epoch_time), #Event generation time in epoch
                                "nodeUniqueId" : id_name, #content id
                                "operationType": "UPDATE", #default to UPDATE
                                "nodeType": "DATA_NODE", #default to DATA_NODE
                                "graphId": domain, #default to domain
                                "objectType": object_type, #object type - content, worksheet, textbook, collection etc
                                "nodeGraphId": 0, #default to 0
                                "transactionData" : {
                                    "properties" : {
                                        "tags": {
                                            "system_contentType": type_of_url, #can be "youtube", "ecml", "pdf"
                                            "system_medium": language_detection(text), #generated using google language detection api
                                            "duration": {
                                                "video" : "", #video duration in seconds
                                                "stage" : ""#can be derived from usage data
                                            },
                                            "num_stage" : num_of_stages, #pdf: number of pages, ecml:number of stages, video:1
                                            "system_plugins" : plugin_used, #id's of plugin used in Content
                                            "system_templates" : template, #id's of templates used in Content
                                            "text": text,
                                            },
                                        "version": pdata, #yaml version
                                        "uri": "" #git commit id
                                         }
                                     }
                                 }
        with open(os.path.join(path_to_id, "ML_content_info.json"), "w") as info:
            mnt_json_dump = json.dump(
                mnt_output_dict_new, info, indent=4) # sort_keys=True,
            print(mnt_json_dump)
        stop = time.time()
        time_consumed = stop-start
        time_consumed_minutes = time_consumed/60.0
        print("time taken in sec for text enrichment for cid -----> {0} : {1}".format(id_name, time_consumed))
        print("time taken in minutes for text enrichment for cid -----> {0} : {1}".format(id_name, time_consumed_minutes))
        logging.info("MTT_TRANSCRIPT_PATH_CREATED: {0}".format(path_to_transcript))
        logging.info("MTT_CONTENT_ID_READ: {0}".format(id_name))
        logging.info("MTT_STOP_FOR_URL {0}".format(url))
        return os.path.join(path_to_id, "ML_content_info.json")
    except BaseException:
            logging.info("TextEnrichment failed for url:{0} with id:{1}".format(url, id_name))


def tagme_text(text, tagme_cred):
    try:
        url = "https://tagme.d4science.org/tagme/tag"

        querystring = {
            "lang": "en",
            "include_categories": False,
            "gcube-token": tagme_cred['gcube_token'],
            "text": "test"}
        headers = {
            'cache-control': "no-cache" 
        }
        response = requests.request(
            "GET",
            url,
            headers=headers,
            params=querystring).json()
        assert response['annotations'][0]["spot"] == "test"
        connection_status = True 
    except ConnectionError:
        print("Unable to establish connection with Tagme")
    if connection_status: 
        try: 
            url = "https://tagme.d4science.org/tagme/tag"
            querystring = {
                "lang": "en",
                "include_categories": True,
                "gcube-token": tagme_cred['gcube_token'],
                "text": text}
            headers = {
                'cache-control': "no-cache",
                'postman-token': tagme_cred['postman_token'] 
            }
            response = requests.request(
                "GET",
                url,
                headers=headers,
                params=querystring).json()
            df = pd.DataFrame(response['annotations'])
        except:
            print("Tagme Failed")
    return df

def run_tagme(path_to_text, tagme_cred):
    file_ = open(path_to_text, "r")
    text = file_.readline()
    words = text.split(" ")
    index_count = 0
    window_len = 700
    response_list = []
    while index_count < len(words):
        text = ' '.join(words[index_count:min(
            (index_count + window_len - 1), len(words))])
        index_count += window_len
        if text:
            response_list.append(tagme_text(text, tagme_cred))
        response_df = pd.concat(response_list)
        response_df.reset_index(drop=True, inplace=True)
    return response_df


def get_tagme_spots(path_to_text, tagme_cred):
    file_ = open(path_to_text, "r")
    text = file_.readline()
    # text = text.encode('utf-8').decode('ascii', 'ignore')
    words = text.split(" ")
    index_count = 0
    window_len = 700
    response_list = []
    while index_count < len(words):
        text = ' '.join(words[index_count:min(
            (index_count + window_len - 1), len(words))])
        index_count += window_len
        response_list.append(tagme_text(text, tagme_cred))
        response_df = pd.concat(response_list)
        try:
            response_df = response_df.drop_duplicates('spot')
            response_df.reset_index(drop=True, inplace=True)
            cleaned_keyword_list = [str(x).lower() for x in list(
                set(response_df['spot'])) if str(x) != 'nan']
            cleaned_keyword_list = clean_string_list(cleaned_keyword_list)
            unique_cleaned_keyword_list = list(set(cleaned_keyword_list))
            spot = pd.DataFrame(unique_cleaned_keyword_list, columns=['keyword'])
        except:
            print("No keywords identified")
            spot =  pd.DataFrame(columns=['keyword'])
    return spot


def text_token_taxonomy_intersection_keywords(
        text_df,
        taxonomy_keywords_set):
    try:
        token = [i.lower() for i in list(text_df['keyword'])]
        common_words = list(set(taxonomy_keywords_set) & set(token))
        text_tax_df = pd.DataFrame(common_words, columns=['keyword'])
        return text_tax_df
    except BaseException:
        logging.info("Keywords cannot be extracted")


def tagme_taxonomy_intersection_keywords(
        tagme_df,
        taxonomy_keywords_set,
        ):
    try:
        spots = [i.lower() for i in list(tagme_df['keyword'])]
        common_words = list(set(taxonomy_keywords_set) & set(spots))
        tagme_tax_df = pd.DataFrame(common_words, columns=['keyword'])
        return tagme_tax_df
    except BaseException:
        logging.info("Keywords cannot be extracted")


def getTaxonomy(DBpedia_cat):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery("select ?value where { <http://dbpedia.org/resource/Category:"+DBpedia_cat.replace(" ","_")+"> skos:broader{0,4} ?value }")
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        dbpedia_prefix_cat = []
        for result in results["results"]["bindings"]:
            dbpedia_prefix_cat.append(str((result['value']['value'])))
    except BaseException:
        dbpedia_prefix_cat = []
    return(dbpedia_prefix_cat)


def checkSubject(dbpedia_prefix_cat, subject):
    subject_paths = ['http://dbpedia.org/resource/Category:'+i for i in subject]
    return int(bool(sum([1 if i in dbpedia_prefix_cat else 0 for i in subject_paths])))


def checkSubjectPartial(dbpedia_prefix_cat, subject):
    subject_paths = ['http://dbpedia.org/resource/Category:'+i for i in subject]
    return int(bool(sum([int(any([1 if ((i in j) or (j in i)) else 0 for j in dbpedia_prefix_cat])) for i in subject_paths])))


def keyword_filter(tagme_response_df, cache_cred, path_to_category_lookup, subject, update_corpus, filter_score_val, num_keywords):
    print("subject:", subject)
    try:
        for i in ['port', 'host', 'password']:
            assert i in list(cache_cred.keys())
            cache_status=True 
    except:
        try:
            r=redis.Redis(host=cache_cred['host'], port=cache_cred['port'])
            r.set("test","test")
            cache_cred["password"]=""
            cache_status=True 
            print("Trying to establish redis connection without authentication")
        except IOError:
            print("Unable to establish connection with redis cache.")

    keyword_df = pd.DataFrame({'keyword': [], 'dbpedia_score': []})
    if cache_status:
        for ind in range(len(tagme_response_df)):
            keyword = tagme_response_df['spot'][ind]
            score=getRediskey(subject+"."+keyword, cache_cred['host'], cache_cred['port'], cache_cred['password'])
            if score:
                score_df = pd.DataFrame({'keyword': [keyword], 'dbpedia_score': [score]})
            else:
                with open(path_to_category_lookup, 'r') as stream:
                    subject_ls = yaml.load(stream)[subject]
                dbpedia_categories = tagme_response_df['dbpedia_categories'][ind]
                count = 0
                try:
                    for cat in dbpedia_categories:
                        dbpedia_prefix_cat = getTaxonomy(cat)
                        status = checkSubject(dbpedia_prefix_cat, subject_ls)
                        count += status
                    if len(dbpedia_categories) > 0:
                        relatedness = float(count)/float(len(dbpedia_categories))
                    else:
                        relatedness = 0
                except BaseException:
                    relatedness = 0
                score_df = pd.DataFrame({'keyword': [keyword], 'dbpedia_score': [relatedness]})
            keyword_df = keyword_df.append(score_df, ignore_index=True)

    # preprocessing
    keyword_df['keyword'] = [str(x).lower() for x in list((keyword_df['keyword'])) if str(x) != 'nan']
    if update_corpus:
            corpus_update_df = keyword_df.drop_duplicates('keyword')
            corpus_update_df = corpus_update_df.dropna()
            for ind,val in corpus_update_df.iterrows():
                    setRediskey(subject+"."+val['keyword'], val['dbpedia_score'], cache_cred['host'], cache_cred['port'], cache_cred['password'])          
    if filter_score_val:
        try:
            keyword_df = keyword_df[keyword_df['dbpedia_score'] >= float(filter_score_val)]  ### from yaml#filtered_keyword_df
        except BaseException:
            print("Error: Invalid filter_score_val. Unable to filter. ")
    if num_keywords:
        try:
            keyword_df = keyword_df.sort_values('dbpedia_score', ascending=[False]).iloc[0:int(num_keywords)]
        except BaseException:
            print("Error: Invalid num_keywords. Unable to filter. ")
    # keyword_relatedness_df.iloc[0:4]['KEYWORDS'].to_csv(Path_to_keywords + "KEYWORDS.csv")
    return keyword_df


def keyword_extraction_parallel(
        dir,
        timestr,
        content_to_text_path,
        extract_keywords,
        filter_criteria,
        cache_cred,
        path_to_category_lookup,
        update_corpus,
        filter_score_val,
        num_keywords,
        tagme_cred):
    """
    A custom function to parallelly extract keywords
    for all the Content texts in a folder.
    This is run typically after multimodal_text_enrichment.
    Part of Content enrichment pipeline.
    The funtion allows keyword extraction using TAGME or
    tokenising the words using nltk tokeniser.
    The extracted keywords can be filtered based on following criteria:
        - taxonomy: if the word is a taxonomy keyword
        - dbpedia: if the keyword is domain keyword based on dbpedia criteria
        (occurs under the domain ontolgy in wikipedia)
        - none
    :param dir(str): Name of the folder containing enriched_text.txt file inside it.
    :param content_to_text_path(str): Path to directory containing multiple Content id folders.
    :param taxonomy(str): Path to taxonomy file(csv)
    :param extract_keywords(str): can be ``tagme`` or ``text_token``
    :param filter_criteria(str): can be ``taxonomy`` or ``none``
    :returns: Path to extracted keywords.
    """
    print("*******dir*********:", dir)
    print("***Extract keywords***:", extract_keywords)
    print("***Filter criteria:***", filter_criteria)
    path_to_id = os.path.join(content_to_text_path, dir)
    content_info_json_loc = os.path.join(path_to_id, "ML_content_info.json")
    if os.path.exists(content_info_json_loc):
        with open(content_info_json_loc, "r") as json_loc:
            content_info = json.load(json_loc)
        #subject = content_info["domain"]
        subject = content_info["graphId"]
    else:
        subject = "none"
    logging.info("Subject of the id: {0}".format(subject))
    path_to_cid_transcript = os.path.join(
        path_to_id, "enriched_text.txt")
    path_to_saved_keywords = ""
    path_to_keywords = os.path.join(
        content_to_text_path, dir, "keywords", extract_keywords + "_" + filter_criteria)
    if os.path.isfile(path_to_cid_transcript):
        logging.info("Transcript present for cid: {0}".format(dir))
        #try:
        if os.path.getsize(path_to_cid_transcript) > 0:
            os.makedirs(path_to_keywords)
            print("Path to transcripts ", path_to_cid_transcript)
            print("Running keyword extraction for {0}".format(
                path_to_cid_transcript))
            print("---------------------------------------------")

            if extract_keywords == "tagme" and filter_criteria == "dbpedia":
                print("Tagme keyword extraction is running for {0}".format(
                    path_to_cid_transcript))
                tagme_response_df = run_tagme(path_to_cid_transcript, tagme_cred)
                keyword_filter_df = keyword_filter(tagme_response_df, cache_cred, path_to_category_lookup, subject, update_corpus, filter_score_val, num_keywords)
                path_to_saved_keywords = os.path.join(path_to_keywords, "keywords.csv")
                print("keyword_filter_df:", keyword_filter_df)
                keyword_filter_df.to_csv(path_to_saved_keywords, index=False, encoding='utf-8')

            elif extract_keywords == "tagme" and filter_criteria == "none":
                print("Tagme keyword extraction is running for {0}".format(
                    path_to_cid_transcript))
                tagme_df = get_tagme_spots(path_to_cid_transcript, tagme_cred)
                path_to_saved_keywords = os.path.join(path_to_keywords, "keywords.csv")
                tagme_df.to_csv(path_to_saved_keywords, index=False, encoding='utf-8')
                logging.info(
                    "Path to tagme tokens is {0}".format(path_to_saved_keywords))

            elif extract_keywords == "tagme" and filter_criteria == "taxonomy":
                print("Tagme intersection taxonomy keyword extraction is running for {0}".format(
                    path_to_cid_transcript))
                clean_keywords = map(get_words, list(taxonomy["Keywords"]))
                clean_keywords = map(clean_string_list, clean_keywords)
                flat_list = [item for sublist in list(
                    clean_keywords) for item in sublist]
                taxonomy_keywords_set = set([clean_text(i) for i in flat_list])
                tagme_df = get_tagme_spots(path_to_cid_transcript, tagme_cred)
                path_to_saved_keywords = os.path.join(path_to_keywords, "keywords.csv")
                tagme_taxonomy_df = tagme_taxonomy_intersection_keywords(
                    tagme_df, taxonomy_keywords_set)
                tagme_taxonomy_df.to_csv(path_to_saved_keywords, index=False, encoding='utf-8')
                print("Path to tagme taxonomy intersection tokens is {0}".format(
                    path_to_saved_keywords))

            elif extract_keywords == "text_token" and filter_criteria == "none":
                print("Text tokens extraction running for {0}".format(
                    path_to_cid_transcript))
                text_df = get_tokens(
                    path_to_cid_transcript)
                path_to_saved_keywords = os.path.join(path_to_keywords, "keywords.csv")
                text_df.to_csv(path_to_saved_keywords, index=False, encoding='utf-8')
                print("Path to text tokens is {0}".format(
                    path_to_saved_keywords))

            elif extract_keywords == "text_token" and filter_criteria == "taxonomy":
                print("Text tokens intersection taxonomy running for {0}".format(
                    path_to_cid_transcript))
                text_df = get_tokens(
                    path_to_cid_transcript)
                clean_keywords = map(get_words, list(taxonomy["Keywords"]))
                clean_keywords = map(clean_string_list, clean_keywords)
                flat_list = [item for sublist in list(
                    clean_keywords) for item in sublist]
                taxonomy_keywords_set = set([clean_text(i) for i in flat_list])
                text_tax_df = text_token_taxonomy_intersection_keywords(
                    text_df, taxonomy_keywords_set)
                path_to_saved_keywords = os.path.join(path_to_keywords, "keywords.csv")
                text_tax_df.to_csv(path_to_saved_keywords, index=False, encoding='utf-8')
                print("Path to text tokens intersection taxonomy is {0}".format(
                    path_to_saved_keywords))

            else:
                print("Invalid argument provided")

        else:
            print("The text file {0} has no contents".format(
                path_to_cid_transcript))

        # except BaseException:
        #     print("Raise exception for {0} ".format(path_to_cid_transcript))
        #     logging.info("Raise exception for {0} ".format(path_to_cid_transcript))
    else:
        print("Transcripts doesnt exist for {0}".format(
            path_to_cid_transcript))
    if path_to_saved_keywords:
        keywords_dpediaScore = pd.read_csv(path_to_saved_keywords).to_json(orient='records')
    else:
        keywords_dpediaScore = []
    # Reading pdata:
    airflow_home = os.getenv('AIRFLOW_HOME', os.path.expanduser('~/airflow'))
    dag_location = os.path.join(airflow_home, 'dags')
    filename = os.path.join(dag_location, 'graph_location')
    f = open(filename, "r")
    f = open(filename, "r")
    lines = f.read().splitlines()
    pdata = lines[-1]
    f.close()
    print("AIRFLOW_HOME: ", dag_location)
    # estimating ets:
    epoch_time = time.mktime(time.strptime(timestr, "%Y%m%d-%H%M%S"))
    kep_output_dict = { 'ets':  int(epoch_time),
                        'keywords': keywords_dpediaScore,
                        "pdata": pdata,
                        "commit_id": ""
                        }
    kep_output_dict_new = { "ets" : int(epoch_time), #Event generation time in epoch
                            "transactionData" : {
                                "properties" : {
                                    "tags": {
                                        "system_keywords": keywords_dpediaScore,
                                        },
                                    "version": pdata, #yaml version
                                    "uri": ""
                                     }
                                }
                           }

    with open(os.path.join(path_to_id, "ML_keyword_info.json"), "w") as info:
        kep_json_dump = json.dump(
            kep_output_dict_new, info, indent=4) # sort_keys=True,
        print(kep_json_dump)
    return content_to_text_path


def get_level_keywords(taxonomy_df, level):
    level_keyword_df = []
    for subject in list(set(taxonomy_df[level])):
        Domain_keywords = list(
            taxonomy_df.loc[taxonomy_df[level] == subject, 'Keywords'])
        unique_keywords = [
            ind for sublist in Domain_keywords for ind in sublist]
        level_keyword_df.append({level: subject, 'Keywords': unique_keywords})
    level_keyword_df = pd.DataFrame(level_keyword_df)
    return level_keyword_df


def getGradedigits(class_x):
    for i in ["Class", "[", "]", " ", "class", "Grade", "grade"]:
        class_x = class_x.replace(i, "")
    return class_x


def precision_from_dictionary(predicted_df, observed_df, window_len):
    window = range(1, window_len + 1)
    percent_list = []
    for ind in window:
        count = 0
        for cid in predicted_df.index:
            try:
                obs_tags = observed_df.loc[cid].values[0].replace("'", "").split(",")
                pred_tags = list(predicted_df.loc[cid][0:ind])
                if bool(set(obs_tags) & set(pred_tags)):
                    count += 1
            except BaseException:
                print(str(cid) + " metadata not available")
        percent_list.append(count * 100.0 / len(predicted_df.index))
    return pd.DataFrame(percent_list, index=window, columns=["percent"])


def agg_precision_from_dictionary(predicted_dct, observed_dct, window_len):
    predicted_val_list = []
    observed_val_list = []
    for subject in observed_dct.keys():
        predicted_df = predicted_dct[subject]
        for i in range(predicted_df.shape[0]):
            predicted_val_list.append(list(predicted_df.iloc[i]))
            observed_val_list.append(observed_dct[subject].iloc[i][0])
    window = [0] * window_len
    for i in range(0, window_len):
        count = 0
        for ind in range(len(predicted_val_list)):
            if observed_val_list[ind] in predicted_val_list[ind][0:min(
                    len(predicted_val_list[ind]), i + 1)]:
                count += 1
        window[i] = 100.0 * count / len(predicted_val_list)
    return pd.DataFrame(window, columns=['Percent'])

