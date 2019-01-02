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
import xml.etree.ElementTree as ET
from dateutil.parser import parse
from SPARQLWrapper import SPARQLWrapper, JSON
logging.getLogger("requests").setLevel(logging.WARNING)

nltk.download("stopwords")
nltk.download("wordnet")
stopwords = stopwords.words('english')


pun_list = list(string.punctuation)


def language_detection(text):
    """
    This function will take in an enriched text as input and
    use google translate API to detect language of the text and returns it

    Parameters
    ----------
    arg1: type
    description

    Returns
    -------
    result: type
    description
    """
    translate_client = translate.Client()
    result = translate_client.detect_language(text)
    return result["language"]


def is_date(date_string):
    try:
        parse(date_string)
        return True
    except ValueError:
        return False


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
    except BaseException:
        return False


def findFiles(directory, substrings):
    """
    Accio!!
    For a given directory, the function looks for any occurance of a particular
    file type mentioned by the substrings parameter.

    Parameters
    ----------
    directory: str
    path to a folder

    substrings: list
    an array of extentions to be searched within the directory.
    ex: jpg, png, webm, mp4

    Returns
    -------
    list: list of paths to detected files
    """
    ls = []
    if isinstance(directory, str) and isinstance(substrings, list):
        if os.path.isdir(directory):
            for dirname, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    string = os.path.join(dirname, filename)
                    for substring in substrings:
                        if(string.find(substring) >= 0):
                            ls.append(string)
    return ls


def unzip_files(directory):
    assert isinstance(directory, str)
    zip_list = findFiles(directory, ['.zip'])
    bugs = {}
    for zip_file in zip_list:
        try:
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(directory)
            os.remove(zip_file)
        except BaseException:
            bugs.append(zip_file)


def merge_json(merge_json_loc_list):
    ignore_list = ["ETS"]
    dict_list = []
    for file in merge_json_loc_list:
        with open(file, "r", encoding="UTF-8") as info:
            new_json = json.load(info)
            [new_json.pop(ignore) for ignore in ignore_list if ignore in new_json.keys()]
        dict_list.append(new_json)
    merge_dict = {}
    for dictionary in dict_list:
        for k, _ in dictionary.items():
            if k in merge_dict.keys():
                print(k)
                try:
                    assert dictionary.get(k) == merge_dict.get(k)
                    print(merge_dict.get(k))
                    merge_dict.update({k: dictionary.get(k)})
                except BaseException:
                    print("Trying to merge keys with different values")
                    pass
            else:
                merge_dict.update({k: dictionary.get(k)})
    return merge_dict


def ekstep_ecar_unzip(download_location, copy_location):
    """
    This function unzips an ecar file(ekstep file format)
    and parses all the subfolder.
    All the files are copied into one of 'assets','data','items' folder
    (same name as in downloaded folder is maintained)
    based on its location in the downloaded folder.
    ==========
    arguments:
        download_location: A location in the disk where ekstep ecar resource file in  downloaded
        copy_location: A disk location where the ecar is unwrapped
    """
    assert isinstance(download_location, str)
    assert isinstance(copy_location, str)
    if not os.path.exists(copy_location):
        os.makedirs(copy_location)
    if not os.path.exists(copy_location):
        os.makedirs(copy_location)
    location = [os.path.join(copy_location, folder)
                for folder in ['assets', 'data', 'items']]

    for loc in location:
        if not os.path.exists(loc):
            os.makedirs(loc)
    for subfolder in os.listdir(os.path.join(download_location)):
        if os.path.isdir(
            os.path.join(
                download_location,
                subfolder)) and len(
            os.listdir(
                os.path.join(
                    download_location,
                    subfolder))) > 0:
            for file in os.listdir(os.path.join(download_location, subfolder)):
                shutil.copy(
                    os.path.join(
                        download_location, subfolder, file), os.path.join(
                        copy_location, "assets"))
        else:
            shutil.copy(
                os.path.join(
                    download_location,
                    subfolder),
                copy_location)


def download_from_downloadUrl(url_to_download, path_to_folder, file_name):
    download_dir = os.path.join(path_to_folder, 'temp' + file_name)
    status = downloadZipFile(url_to_download, download_dir)
    try:
        if status:
            unzip_files(download_dir)
            ekstep_ecar_unzip(
                download_dir, os.path.join(
                    path_to_folder, file_name))
            shutil.rmtree(download_dir)
            path_to_file = os.path.join(path_to_folder, file_name)
            return path_to_file
    except BaseException:
        print("Unavailable for download")


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
    replace_char = [
        "[",
        "]",
        "u'",
        "None",
        "Thank you",
        "-",
        "(",
        ")",
        "#",
        "Done",
        ">",
        "<",
        "-",
        "|",
        "/",
        "\"",
        "Hint",
        "\n",
        "'"]
    for l in replace_char:
        text = text.replace(l, "")
    text = re.sub(' +', ' ', text)
    return text


def clean_text_tokens(text):
    """
    A custom preprocessor to tokenise and clean a text.
    Used in Content enrichment pipeline.
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
    tokens = [re.sub(r'[0-9\.\W_]', '', token) for token in tokens]
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if len(token) > 1]
    tokens = [token for token in tokens if token]
    return tokens


def strip_word(word, delimitter):
    """
    Replace punctuations from string, punctuation and space in a word
    with a DELIMITTER

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
    delimitters = ["___", "__", " ", ",", "_", "-", ".", "/"] + \
        list(set(string.punctuation))
    for lim in delimitters:
        word = word.replace(lim, delimitter)
    return word


def identify_contentType(url):
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
            return "https://www.youtube.com/watch?v=" + \
                fetch_video_id(url.split("embed/")[1])[:11]
        else:
            return "https://www.youtube.com/watch?v=" + \
                youtube_regex_match.group(6)


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


def audio_to_text(path_to_audio_split_folder, GOOGLE_APPLICATION_CREDENTIALS):
    text = ""
    for i in natsorted(os.listdir(path_to_audio_split_folder), reverse=False):
        if i[-4:] == ".mp3":
            try:
                text += getTexts(os.path.join(path_to_audio_split_folder,
                                              i), 'en-IN', GOOGLE_APPLICATION_CREDENTIALS)['text']
            except LookupError:
                continue
    return text


def text_conversion(path_to_audio_split_folder, path_to_text_folder):
    logging.info("TC_START for audio split folder: {0}". format(
        path_to_audio_split_folder))
    if not os.path.exists(path_to_text_folder):
        os.mkdir(path_to_text_folder)

    path_ = os.path.join(path_to_text_folder, "enriched_text.txt")
    print("type of audio to text: ", type(
        audio_to_text(path_to_audio_split_folder, GOOGLE_APPLICATION_CREDENTIALS)))
    with open(path_, "w") as myTextFile:
        myTextFile.write(audio_to_text(path_to_audio_split_folder, GOOGLE_APPLICATION_CREDENTIALS))

    logging.info("TC_TRANSCRIPT_PATH_CREATED: {0}".format(path_))
    logging.info("TC_STOP for audio split folder: {0}". format(
        path_to_audio_split_folder))
    return path_


def getText_json(jdata, key):
    sdata = json.dumps(jdata)
    text_list = ([sdata[(m.start(0) + len(key) + 4):m.end(0) - 1]
                  for m in re.finditer(key + '": "(.*?)"', sdata)])
    return text_list


def download_to_local(method, url_to_download, path_to_save, id_name):
    logging.info("DTL_START_FOR_URL: {0}".format(url_to_download))
    path_to_id = ""
    if method == "ecml":
        logging.info("DTL_ECAR_URL: {0}".format(url_to_download))
        try:
            path_to_id = download_from_downloadUrl(
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
            path_to_id = download_from_downloadUrl(
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
    video_names = findFiles(path_to_assets, ['mp4'])
    logging.info('...detected {0} video files'.format(str(len(video_names))))
    if method == "ffmpeg" and len(video_names) > 0:
        logging.info("VTS_START_FOR_METHOD: {0}".format(method))

        for file in video_names:
            # ffmpy wrapper to convert mp4 to mp3:
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


# def pdf_to_text(method, path_to_assets, pdf_url):
#     text = ""
#     number_of_pages = 0
#     logging.info("PTT_START")
#     if method == "PyPDF2":
#         logging.info("PTT_METHOD: {0}".format(method))
#         pdf_names = findFiles(path_to_assets, ['.pdf'])
#         text = ""
#         for j in range(0, len(pdf_names) + 1):
#             if (len(pdf_names) == 0 and pdf_url.endswith('pdf')):
#                 r = requests.get(pdf_url)
#                 f = io.BytesIO(r.content)
#                 read_pdf = PdfFileReader(f)
#                 number_of_pages = read_pdf.getNumPages()
#             elif j < (len(pdf_names)):
#                 pdf_files = pdf_names[j]
#                 text = ""
#                 text = convert_pdf_to_txt(pdf_files)
#                 number_of_pages = 0
#                 # f = open(pdf_files, 'rb')
#                 # read_pdf = PdfFileReader(f)
#                 # number_of_pages = read_pdf.getNumPages()
#             else:
#                 number_of_pages = 0
#             if number_of_pages > 0:
#                 for i in range(number_of_pages):
#                     page = read_pdf.getPage(i)
#                     page_content = page.extractText()
#                     text += page_content
#         processed_txt = cleantext(text)
#         text = ''.join([i for i in processed_txt if not i.isdigit()])
#         text = ' '.join(text.split())
#     if method == "none":
#         logging.info("PDF_NOT_PERFORMED")

#     logging.info("PTT_STOP")
#     text_dict = {"text": text, "no_of_pages": number_of_pages}
#     return text_dict


def pdf_to_text(method, path_to_assets, pdf_url):
    text = ""
    number_of_pages = 0
    logging.info("PTT_START")
    pdf_names = findFiles(path_to_assets, ['.pdf'])
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
    processed_txt = cleantext(text)
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


def ecml_index_to_text(method, path_to_id):
    all_text = ""
    plugin_used = []
    num_stages = 0
    logging.info("JTT_START")
    if method == "parse":
        if os.path.exists(os.path.join(path_to_id, "index.ecml")):
            ecml_file = os.path.join(path_to_id, "index.ecml")
            try:
                logging.info('...File type detected as ecml')
                ecml_tags = ecml_parser(ecml_file)
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
        logging.info("JTT_NOT_PERFORMED")
    logging.info("JTT_STOP")
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
    index, content_meta, content_type, content_to_text_path
    A custom function to extract text from a given
    Content id in a Content meta dataframe extracted using Content V2 api

    Parameters
    ----------
    index: intpdf_to
    row id for the Content

    content_meta: dataframe
    A dataframe of Content metadata.
    Mandatory fields: ['artifactUrl', 'content_type','downloadUrl',
    'gradeLevel', 'identifier','keywords', 'language', 'subject']

    content_type: str
    Can be youtube, pdf, ecml, unknown

    content_to_text_path:str
    path to save the extracted text

    Returns
    -------
    path_to_transcript: str
    Path where text is saved
    """
    type_of_url = content_meta.iloc[index]["derived_contentType"]
    id_name = content_meta["identifier"][index]
    downloadField = content_type[type_of_url]["contentDownloadField"]
    url = content_meta[downloadField][index]
    logging.info("MTT_START_FOR_INDEX {0}".format(index))
    logging.info("MTT_START_FOR_CID {0}".format(id_name))
    logging.info("MTT_START_FOR_URL {0}".format(url))
    # start text extraction pipeline:
    # try:
    path_to_id = download_to_local(
        type_of_url, url, content_to_text_path, id_name)
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
    pdata = f.read()
    f.close()

    # estimating ets:
    epoch_time = time.mktime(time.strptime(timestr, "%Y%m%d-%H%M%S"))
    domain = content_meta["subject"][index]

    template = ""
    plugin_used = []
    num_of_stages = 0
    # only for type ecml
    if type_of_url == "ecml":
        plugin_used = ecml_index_to_text("parse", path_to_id)["plugin_used"]
        num_of_stages = ecml_index_to_text("parse", path_to_id)["num_stage"]

    mnt_output_dict = {
                'ETS': int(epoch_time),
                'content_id': id_name,
                'content_type': type_of_url,
                'domain': domain,
                'medium': language_detection(text),
                'duration': duration,
                'plugin_used': plugin_used,
                'num_of_stages': num_of_stages,
                'template': template,
                'text': text,
                'pdata': pdata,
                'commit_id': ""
            }

    with open(os.path.join(path_to_id, "ML_content_info.json"), "w") as info:
        mnt_json_dump = json.dump(
            mnt_output_dict, info, sort_keys=False, indent=4)
        print(mnt_json_dump)
    logging.info("MTT_TRANSCRIPT_PATH_CREATED: {0}".format(path_to_transcript))
    logging.info("MTT_CONTENT_ID_READ: {0}".format(id_name))
    logging.info("MTT_STOP_FOR_URL {0}".format(url))
    #return os.path.join(path_to_id, "ML_content_info.json")
    # except BaseException:
    #     logging.info("TextEnrichment failed for url:{0} with id:{1}".format(url, id_name))


def custom_tokenizer(path_to_text_file):
    """
    Given a text file uses custom_tokenizer function
    to tokenise and write the tokenised words to a keywords.csv file.

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
    text = open(path_to_text_file, "r")
    text_file = text.read()
    text_list = clean_text_tokens(text_file)
    text_df = pd.DataFrame(text_list, columns=['KEYWORDS'])
    return text_df


def tagme_text(text):
    url = "https://tagme.d4science.org/tagme/tag"

    querystring = {
        "lang": "en",
        "include_categories": True,
        "gcube-token": "1e1f2881-62ec-4b3e-9036-9efe89347991-843339462",
        "text": text}

    headers = {
        'gcube-token': "1e1f2881-62ec-4b3e-9036-9efe89347991-843339462",
        'cache-control': "no-cache",
        'postman-token': "98279373-78af-196e-c040-46238512c338"
    }

    response = requests.request(
        "GET",
        url,
        headers=headers,
        params=querystring).json()
    df = pd.DataFrame(response['annotations'])
    return df


def run_tagme(path_to_text):
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
            response_list.append(tagme_text(text))
        response_df = pd.concat(response_list)
        response_df.reset_index(drop=True, inplace=True)
    return response_df


def get_tagme_spots(path_to_text):
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
        response_list.append(tagme_text(text))
        response_df = pd.concat(response_list)
        response_df = response_df.drop_duplicates('spot')
        response_df.reset_index(drop=True, inplace=True)
        cleaned_keyword_list = [str(x).lower() for x in list(
            set(response_df['spot'])) if str(x) != 'nan']
        cleaned_keyword_list = clean_string_list(cleaned_keyword_list)
        unique_cleaned_keyword_list = list(set(cleaned_keyword_list))
        spot = pd.DataFrame(unique_cleaned_keyword_list, columns=['keyword'])
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


def keyword_filter(tagme_response_df, path_to_corpus, path_to_category_lookup, subject, update_corpus, filter_score_val, num_keywords):
    print("subject:", subject)

    corpus_lookup_filename = os.path.join(path_to_corpus, subject + ".csv")
    print("Lookup corpus :"+corpus_lookup_filename)
    if not os.path.exists(os.path.dirname(corpus_lookup_filename)):
        try:
            print("Creating :", os.path.dirname(corpus_lookup_filename))
            os.makedirs(os.path.dirname(corpus_lookup_filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    if os.path.isfile(corpus_lookup_filename):
        lookup_df = pd.read_csv(os.path.join(path_to_corpus, subject + ".csv"))
    else:
        lookup_df = pd.DataFrame({'keyword': [], 'dbpedia_score': []})
        lookup_df.to_csv(corpus_lookup_filename)
    keyword_df = pd.DataFrame({'keyword': [], 'dbpedia_score': []})
    for ind in range(len(tagme_response_df)):
        keyword = tagme_response_df['spot'][ind]

        if keyword in list(lookup_df['keyword']):
            lookup_df = lookup_df[lookup_df['keyword'] == keyword]
            relatedness = lookup_df.iloc[0]['dbpedia_score']
            score_df = pd.DataFrame({'keyword': [keyword], 'dbpedia_score': [relatedness]})
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
            with open(corpus_lookup_filename, 'a') as f:
                corpus_update_df.to_csv(f, header=False, index=False)
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
        taxonomy,
        extract_keywords,
        filter_criteria,
        path_to_corpus,
        path_to_category_lookup,
        update_corpus,
        filter_score_val,
        num_keywords):
    """
    A custom function to parallelly extract keywords
    for all the Content texts in a folder.
    This is run typically after multimodal_text_enrichment.
    Part of Content enrichment pipeline.
    The funtion allows keyword extraction using TAGME or
    tokenising the words using nltk tokeniser.
    The extracted keywords can be filtered based on following criteris:
        -taxonomy: if the word is a taxonomy keyword
        -dbpedia: if the keyword is domain keyword based on dbpedia criteria
        (occurs under the domain ontolgy in wikipedia)
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
    path_to_id = os.path.join(content_to_text_path, dir)
    content_info_json_loc = os.path.join(path_to_id, "ML_content_info.json")
    if os.path.exists(content_info_json_loc):
        with open(content_info_json_loc, "r") as json_loc:
            content_info = json.load(json_loc)
        subject = content_info["domain"]
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
        try:
            if os.path.getsize(path_to_cid_transcript) > 0:
                os.makedirs(path_to_keywords)
                print("Path to transcripts ", path_to_cid_transcript)
                print("Running keyword extraction for {0}".format(
                    path_to_cid_transcript))
                print("---------------------------------------------")

                if extract_keywords == "tagme" and filter_criteria == "dbpedia":
                    print("Tagme keyword extraction is running for {0}".format(
                        path_to_cid_transcript))
                    tagme_response_df = run_tagme(path_to_cid_transcript)
                    keyword_filter_df = keyword_filter(tagme_response_df, path_to_corpus, path_to_category_lookup, subject, update_corpus, filter_score_val, num_keywords)
                    path_to_saved_keywords = os.path.join(path_to_keywords, "keywords.csv")
                    print("keyword_filter_df:", keyword_filter_df)
                    keyword_filter_df.to_csv(path_to_saved_keywords, index=False, encoding='utf-8')

                elif extract_keywords == "tagme" and filter_criteria == "none":
                    print("Tagme keyword extraction is running for {0}".format(
                        path_to_cid_transcript))
                    tagme_df = get_tagme_spots(path_to_cid_transcript)
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
                    taxonomy_keywords_set = set([cleantext(i) for i in flat_list])
                    tagme_df = get_tagme_spots(path_to_cid_transcript)
                    path_to_saved_keywords = os.path.join(path_to_keywords, "keywords.csv")
                    tagme_taxonomy_df = tagme_taxonomy_intersection_keywords(
                        tagme_df, taxonomy_keywords_set)
                    tagme_taxonomy_df.to_csv(path_to_saved_keywords, index=False, encoding='utf-8')
                    print("Path to tagme taxonomy intersection tokens is {0}".format(
                        path_to_saved_keywords))

                elif extract_keywords == "text_token" and filter_criteria == "none":
                    print("Text tokens extraction running for {0}".format(
                        path_to_cid_transcript))
                    text_df = custom_tokenizer(
                        path_to_cid_transcript)
                    path_to_saved_keywords = os.path.join(path_to_keywords, "keywords.csv")
                    text_df.to_csv(path_to_saved_keywords, index=False, encoding='utf-8')
                    print("Path to text tokens is {0}".format(
                        path_to_saved_keywords))

                elif extract_keywords == "text_token" and filter_criteria == "taxonomy":
                    print("Text tokens intersection taxonomy running for {0}".format(
                        path_to_cid_transcript))
                    text_df = custom_tokenizer(
                        path_to_cid_transcript)
                    clean_keywords = map(get_words, list(taxonomy["Keywords"]))
                    clean_keywords = map(clean_string_list, clean_keywords)
                    flat_list = [item for sublist in list(
                        clean_keywords) for item in sublist]
                    taxonomy_keywords_set = set([cleantext(i) for i in flat_list])
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

        except BaseException:
            print("Raise exception for {0} ".format(path_to_cid_transcript))
            logging.info("Raise exception for {0} ".format(path_to_cid_transcript))
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
    pdata = f.read()
    f.close()
    print("AIRFLOW_HOME: ", dag_location)
    # estimating ets:
    epoch_time = time.mktime(time.strptime(timestr, "%Y%m%d-%H%M%S"))
    kep_output_dict = {
                        'keywords': keywords_dpediaScore,
                        'content_id': dir,
                        "pdata": pdata,
                        "commit_id": "",
                        "ETS": int(epoch_time)
                        }

    with open(os.path.join(path_to_id, "ML_keyword_info.json"), "w") as info:
        kep_json_dump = json.dump(
            kep_output_dict, info, sort_keys=True, indent=4)
        print(kep_json_dump)
    return content_to_text_path


def clean_string_list(x_list):
    x_list = list(map((str.lower), x_list))
    x_clean = [i.lstrip() for i in x_list]
    x_clean = [i.rstrip() for i in x_clean]
    x_clean = list(filter(None, x_clean))
    return x_clean


def get_words(x):
    if str(x) != 'nan':
        x = x.split(', ')
        return x
    else:
        return ""


def custom_listPreProc(key_list, preproc, DELIMITTER):
    key_list = [clean_string_list(x) for x in key_list]
    key_list_clean = []
    for x in key_list:
        x = [strip_word(i, DELIMITTER) for i in x]
        if preproc == 'stem_lem':
            key_list_clean.append(stem_lem((x), DELIMITTER))
        else:
            print("unknown preproc")
            return
    return key_list_clean


def stem_lem(keyword_list, DELIMITTER):
    wordnet_lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    keyword_list = [item for item in keyword_list]
    keyword_list = [i.split(DELIMITTER) for i in keyword_list]
    lemma_ls_1 = [[wordnet_lemmatizer.lemmatize(
        item, pos="n") for item in words] for words in keyword_list]
    lemma_ls_2 = [[wordnet_lemmatizer.lemmatize(
        item, pos="v") for item in words] for words in lemma_ls_1]
    lemma_ls_3 = [[wordnet_lemmatizer.lemmatize(
        item, pos="a") for item in words] for words in lemma_ls_2]
    lemma_ls_4 = [[wordnet_lemmatizer.lemmatize(
        item, pos="r") for item in words] for words in lemma_ls_3]
    stemm_ls = [[stemmer.stem(item) for item in words] for words in lemma_ls_4]
    return [DELIMITTER.join(i) for i in stemm_ls]


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


def match_str_list(list1, list2):
    intersection = 1.0 * (len(set(list1) & set(list2)))
    union = 1.0 * (len(set(list1 + list2)))
    if union != 0:
        jaccard_index = intersection / union
    else:
        jaccard_index = 0
    try:
        cosine_similarity = intersection / (len(set(list1)) * len(set(list2)))
    except BaseException:
        cosine_similarity = 0
    if len(set(list1)) != 0:
        match_percent_l1 = float(intersection) / len(set(list1))
    else:
        match_percent_l1 = 0
    if len(set(list2)) != 0:
        match_percent_l2 = float(intersection) / len(set(list2))
    else:
        match_percent_l2 = 0
    return {
        'intersection': intersection,
        'union': union,
        'jaccard_index': jaccard_index,
        'cosine_similarity': cosine_similarity,
        'match_percent_l1': match_percent_l1,
        'match_percent_l2': match_percent_l2}


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


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin1')


def getPhrase(x_list, DELIMITTER):
    # x_list=clean_string_list(x_list)
    x_phrase = [i for i in x_list if DELIMITTER in i]
    x_word = [item for item in x_list if item not in x_phrase]
    return x_word, x_phrase


def removeShortWords(mylist, wordlen):
    return [item for item in mylist if len(item) > wordlen]


def WordtoPhraseMatch(wordlist, phraselist, DELIMITTER):
    phrasewords = [item.split(DELIMITTER) for item in phraselist]
    match_count = 0
    partial_match_list = []
    wordlist_dynamic = wordlist[:]
    for items in phrasewords:
        word_count = 0
        wordlist = wordlist_dynamic[:]
        for word in wordlist:
            if word in items:
                word_count += 1
                partial_match_list.append((word, DELIMITTER.join(items)))
                wordlist_dynamic.remove(word)
        match_count += int(bool(word_count))
    return partial_match_list, match_count


def jaccard_with_phrase(list1, list2):
    DELIMITTER = "_"
    intersection_words = []
    list1 = list(set(list1))
    list2 = list(set(list2))
    list1 = removeShortWords(list1, 0)
    list2 = removeShortWords(list2, 0)
    list1_words, list1_phrases = getPhrase(list1, DELIMITTER)
    list2_words, list2_phrases = getPhrase(list2, DELIMITTER)
    intersection = 0
    match_count = 0
    # count matching words
    exact_word_intersection = list(set(list1_words) & set(list2_words))
    intersection_words.extend([(a, a) for a in exact_word_intersection])
    exact_word_match = match_str_list(list1_words, list2_words)['intersection']
    intersection = intersection + exact_word_match
    match_count += exact_word_match
    phraselist1 = list1_phrases
    phraselist2 = list2_phrases
    exact_phrase_intersection = []
    for phrase1 in phraselist1:
        for phrase2 in phraselist2:
            if((phrase2 in phrase1) or (phrase1 in phrase2)):
                exact_phrase_intersection.append((phrase1, phrase2))
                list2_phrases.remove(phrase2)
                break
    exact_phrase_length = sum(
        [min([(len(j.split(DELIMITTER))) for j in i]) for i in exact_phrase_intersection])
    intersection += (2.0 * exact_phrase_length)
    match_count += len(exact_phrase_intersection)
    intersection_words.extend(exact_phrase_intersection)
    non_matched_list1_words, non_matched_list2_words = list1_words, list2_words
    non_matched_list1_phrases, non_matched_list2_phrases = list1_phrases, list2_phrases
    if exact_word_intersection:
        non_matched_list1_words = [
            item for item in list1_words if str(item) not in exact_word_intersection]
        non_matched_list2_words = [
            item for item in list2_words if str(item) not in exact_word_intersection]
    if exact_phrase_intersection:
        non_matched_list1_phrases = [
            word for item in exact_phrase_intersection for word in non_matched_list1_phrases if item[0] not in word]
        non_matched_list2_phrases = [
            word for item in exact_phrase_intersection for word in non_matched_list2_phrases if item[1] not in word]
    partial_match_list1, count = WordtoPhraseMatch(
        non_matched_list1_words, non_matched_list2_phrases, DELIMITTER)
    match_count += count
    if partial_match_list1:
        non_matched_list1_words = [
            word for item in partial_match_list1 for word in non_matched_list1_words if item[0] not in word]
        non_matched_list2_phrases = [
            word for item in partial_match_list1 for word in non_matched_list2_phrases if item[1] not in word]
    intersection = intersection + len(partial_match_list1)
    intersection_words.extend(partial_match_list1)
    # Content phrase to taxonomy words
    partial_match_list2, count = WordtoPhraseMatch(
        non_matched_list2_words, non_matched_list1_phrases, DELIMITTER)
    match_count += count
    non_matched_list2_words = [
        item[0] for item in partial_match_list2 if item[0] not in non_matched_list1_phrases]
    intersection = intersection + len(partial_match_list2)
    intersection_words.extend(partial_match_list2)
    intersection_words = [el for el in intersection_words if el != []]

    if (((len(list2)) != 0) & ((len(list1) + len(list2) - match_count) != 0)):
        return {'jaccard': float(intersection) / float(len(list1) + len(list2) - match_count),
                'match_percentage': float(intersection) / float(len(list2)),
                'word_intersection': intersection_words}
    elif ((len(list1) + len(list2) - match_count) == 0):

        return {'jaccard': 0,
                'match_percentage': float(intersection) / float(len(list2)),
                'word_intersection': intersection_words}
    elif (len(list2) == 0):
        return {
            'jaccard': float(intersection) / float(
                len(list1) + len(list2) - match_count),
            'match_percentage': 0,
            'word_intersection': intersection_words}


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


def CustomDateFormater(**kwargs):
    import datetime
    from datetime import date, timedelta
    expected_args = ['x','fileloc', 'datepattern']
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