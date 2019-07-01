import zipfile
import requests
import os
import io
import pickle


def downloadZipFile(url, directory):

    """
    Multimedia Content are stored in cloud in ecar or zip format.
    This function downloads a zip file pointed by url location.
    The user is expected to have access to the file pointed by url.
    The extracted file is available in location specified by directory.
    :param url(str): A valid url pointing to zipped Content location on cloud
    :returns: Status of download.``True``for uccessful download and ``False`` for unsuccesful download
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
    :param directory(str): The path to a folder
    :param substrings(list of strings): An array of extensions to be searched within the directory.
                                        ``eg: jpg, png, webm, mp4``
    :returns: List of paths to the detected files
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
    """
    This function iterates through all the files in a directory and unzip those that are zipped (.zip) to that same folder
    :param directory(str): A directory or path to a folder
    """
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


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin1')
