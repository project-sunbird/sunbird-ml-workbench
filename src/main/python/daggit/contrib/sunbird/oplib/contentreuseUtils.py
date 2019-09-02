import os
import daggit
import requests
import io
import re
import shutil
import json
import numpy as np
import pandas as pd
import ruptures as rp
import Levenshtein

from daggit.core.io.files import findFiles

from google.cloud import vision
from google.cloud import storage
from google.protobuf import json_format
from natsort import natsorted
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
from pyemd import emd


    
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the storage bucket."""
    storage_client = storage.Client()
    gcs_source_uri  = ""
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        gcs_source_uri = "gs://{0}/{1}".format(bucket_name, destination_blob_name)
        print('File {} uploaded to {}.'.format(source_file_name, destination_blob_name))
    except:
        print("Bucket name doesnot exist")
    return gcs_source_uri


def do_GoogleOCR(gcs_source_uri, gcs_destination_uri): #bs parameter
    """
    Perform OCR on a PDF uploaded in google cloud storage, generate output as 
    JSON responses and save it in a destination URI
    """
    # Supported mime_types are: 'application/pdf' and 'image/tiff'
    mime_type = 'application/pdf'

    # How many pages should be grouped into each json output file.
    batch_size = 1

    client = vision.ImageAnnotatorClient()

    feature = vision.types.Feature(
        type=vision.enums.Feature.Type.DOCUMENT_TEXT_DETECTION)

    gcs_source = vision.types.GcsSource(uri=gcs_source_uri)
    input_config = vision.types.InputConfig(
        gcs_source=gcs_source, mime_type=mime_type)

    gcs_destination = vision.types.GcsDestination(uri=gcs_destination_uri)
    output_config = vision.types.OutputConfig(
        gcs_destination=gcs_destination, batch_size=batch_size)

    async_request = vision.types.AsyncAnnotateFileRequest(
        features=[feature], input_config=input_config,
        output_config=output_config)

    operation = client.async_batch_annotate_files(requests=[async_request])
    print('Waiting for the operation to finish.')
    operation.result(timeout=180)

    # Once the request has completed and the output has been
    # written to GCS, we can list all the output files.
    storage_client = storage.Client()

    match = re.match(r'gs://([^/]+)/(.+)', gcs_destination_uri)
    bucket_name = match.group(1)
    print("bucket_name", bucket_name)
    prefix = match.group(2)

    bucket = storage_client.get_bucket(bucket_name)#1.16.0(bucket_or_name=bucket_name)

    # List objects with the given prefix.
    blob_list = list(bucket.list_blobs(prefix=prefix))
    new_list = []
    for i in range(len(blob_list)): 
        str_convert = str(blob_list[i]).replace("<", "").replace(">", "").split(", ")[1]
        if str_convert[-3:] == "pdf":
            pass
        else:
            new_list.append(str_convert)
    sorted_blob_list = [bucket.blob(i) for i in natsorted(new_list, reverse=False)]
    all_text = ""
    #sort the blob_list
    for i in range(len(sorted_blob_list)):
        try:
            output = sorted_blob_list[i]
            json_string = output.download_as_string()
            response = json_format.Parse(
                json_string, vision.types.AnnotateFileResponse())
            first_page_response = response.responses[0]
            annotation = first_page_response.full_text_annotation

        except:
            print("SKIP---->", i)
        all_text += annotation.text

    return prefix[:-1], all_text


def download_outputjson_reponses(bucket_name, prefix, folder_to_save, delimiter=None):
    """Lists all the blobs in the bucket that begin with the prefix.
    This can be used to list all blobs in a "folder", e.g. "public/".
    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned. For example, given these blobs:
        /a/1.txt
        /a/b/2.txt
    If you just specify prefix = '/a', you'll get back:
        /a/1.txt
        /a/b/2.txt
    However, if you specify prefix='/a' and delimiter='/', you'll get back:
        /a/1.txt
    """
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix,
                                      delimiter=delimiter)
    json_saved_in = ""
    for blob in blobs:
        try:
            json_saved_in = "{}/{}".format(folder_to_save, blob.name)
            root_path = os.path.split(json_saved_in)[0]
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            blob.download_to_filename(json_saved_in)
        except:
            print("Unable to download the blob")
    return folder_to_save


def custom_googleOCR_parser(path_to_outputjson_folder):
    json_list = os.listdir(path_to_outputjson_folder)
    if '.DS_Store' in json_list:
        json_list.remove('.DS_Store')
    list_of_outputjson = [os.path.join(path_to_outputjson_folder, i) for i in natsorted(json_list, reverse=False)]
    try:
        for no_ in range(len(list_of_outputjson)):
            with open(list_of_outputjson[no_], "rb") as json_file:
                doc = json.load(json_file)

            document = doc['responses']
            page_boxes = []
            page_txt = []
            page_sizes = []
            for sid, sheet in enumerate(document):
                pages = sheet['fullTextAnnotation']['pages']
                print('sheet: ', sid)
                for pid, page in enumerate(pages):
                    size = [page['width'],page['height']]
                    page_sizes.append(size)
                    boxes = []
                    mytxt = []

                    for bid, block in enumerate(page['blocks']):
                        bbox = block['boundingBox']

                        paragraphs = block['paragraphs']
                        rPara = ''
                        for paid,para in enumerate(paragraphs):
                            bbox = para['boundingBox']
                            rSent = ''
                            for wid, word in enumerate(para['words']):
                                symbols = word['symbols']
                                rWord = ''
                                for sid, symbol in enumerate(symbols):
                                    txt = (symbol['text'].encode("utf-8")).decode("utf-8")
                                    rWord+=txt
                                rSent += ' '+rWord
                            rPara += ''+rSent
                            mytxt.append(rSent)
                            boxes.append(getBox(bbox,size))
                    page_boxes.append(boxes)
                    page_txt.append(mytxt)
        return page_txt
    except:
        print("Parsing unsuccessful")


def getblob(method_of_ocr, bucket_name, local_path_to_pdf, content_id, root_path):
    path_to_outputjson_folder = ""
    if method_of_ocr == "GOCR":
        print("----Performing GoogleOCR Text extraction----")
        try:
            pdf_name = content_id # os.path.split(local_path_to_pdf)[1][:-4]
            textbook_model_path = os.path.join(root_path, pdf_name)
            print(pdf_name, textbook_model_path)
            if not os.path.exists(textbook_model_path):
                os.makedirs(textbook_model_path)
            location = [os.path.join(textbook_model_path, folder)
                        for folder in ['source', 'extract', 'raw_data']]
            for loc in location:
                if not os.path.exists(loc):
                    os.makedirs(loc)
            shutil.copy(local_path_to_pdf, os.path.join(textbook_model_path, "source"))
            gcs_source_uri = upload_blob(bucket_name, local_path_to_pdf, pdf_name+".pdf")
            if gcs_source_uri:
                # perform GoogleOCR:
                gcs_destination_uri = "gs://{0}/{1}".format(bucket_name, os.path.split(gcs_source_uri)[1][:-4]+"/")
                print(gcs_destination_uri)
                prefix, all_text = do_GoogleOCR(gcs_source_uri, gcs_destination_uri)
                path_to_gocr_text = os.path.join(textbook_model_path, "extract", "GOCR", "text")
                path_to_gocr_json = os.path.join(textbook_model_path, "raw_data")
                if not os.path.exists(path_to_gocr_text):
                    os.makedirs(path_to_gocr_text)

                with open(os.path.join(path_to_gocr_text, prefix + ".txt"), "w") as text_file:
                    text_file.write(all_text)
                #concatenate multiple text file if any:
                textnames = findFiles(path_to_gocr_text, ["txt"])
                with open(os.path.join(path_to_gocr_text, "fulltext_annotation" + ".txt"), 'w') as outfile:
                    for fname in textnames:
                        with open(fname) as infile:
                            for line in infile:
                                outfile.write(line)
                            os.remove(fname)
                path_to_outputjson_folder = download_outputjson_reponses(bucket_name, prefix+"/", path_to_gocr_json, delimiter="/")
        except:
           print("Process terminated") 
    return textbook_model_path


def create_manifest(content_id, path_to_saved_folder):
    manifest = {}
    path_to_manifest = ""
    try:
        if os.path.exists(path_to_saved_folder):
            path_to_manifest = os.path.join(path_to_saved_folder, "manifest.json")
            manifest["source"] = {"name": content_id}

            pdf_list = findFiles(path_to_saved_folder, ["pdf"])
            arr = []
            for i in pdf_list:
                arr.append({"id": content_id, "path": i})
            manifest["source"]["path"] = arr
            arr = []
            for i in findFiles(path_to_saved_folder, ["txt"]):
                arr.append({"id": content_id, "path": i, "Type": "gocr"})
                
            manifest["extract"] = {}
            manifest["extract"]["fulltextAnnotation"] = arr
            arr = []
            for i in (os.listdir(os.path.join(path_to_saved_folder, "raw_data"))):
                if i != '.DS_Store':
                    arr.append({"id": content_id+"_blob_gocr", "path": i, "Type": "gocr"})
                
            manifest["extract"]["api_response"] = arr
            with open(path_to_manifest, "w") as json_file:
                json.dump(manifest, json_file, indent=4)
        else:
            print("path doesnot exist!")
    except:
        print("Error in manifest file creation")
    return path_to_manifest


def create_toc(content_id, path_to_saved_folder, api_key, postman_token):

    url = "https://diksha.gov.in/action/composite/v3/search"
    payload = "{\r\n    \"request\": {\r\n        \"filters\":{\r\n            \"identifier\":[\""+content_id+"\"]\r\n         },\r\n               \"limit\":1\r\n    }\r\n}"
    path_to_toc = ""
    headers = {
        'content-type': "application/json",
        'authorization': api_key,
        'cache-control': "no-cache",
        'postman-token': postman_token
        }
    print("path_to_saved_folder", path_to_saved_folder)
    response = requests.request("POST", url, data=payload, headers=headers).json()
    try:
        path_to_toc = os.path.join(path_to_saved_folder, content_id+".json")
        with open(path_to_toc, "w") as write_file:
            json.dump(response["result"]["content"][0], write_file, indent=4)
    except:
        pass
    return path_to_toc
