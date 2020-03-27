import ast
import gc
import json
import logging
import os
import pickle
import re
import shutil
import time
import warnings
from collections import ChainMap
from operator import itemgetter

import gspread
import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import requests
import torch
from daggit.core.io.files import findFiles
from daggit.core.oplib import distanceUtils as dist
from gensim.models import Word2Vec
from google.cloud import storage
from google.cloud import vision
from google.protobuf import json_format
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from natsort import natsorted
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel
from transformers import BertTokenizer

warnings.filterwarnings("ignore")


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the storage bucket."""
    storage_client = storage.Client()
    gcs_source_uri = ""
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        gcs_source_uri = "gs://{0}/{1}".format(bucket_name, destination_blob_name)
        print('File {} uploaded to {}.'.format(source_file_name, destination_blob_name))
    except:
        print("Bucket name doesnot exist")
    return gcs_source_uri


def do_GoogleOCR(gcs_source_uri, gcs_destination_uri):  # bs parameter
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

    bucket = storage_client.get_bucket(bucket_name)

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
    # sort the blob_list
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
                    size = [page['width'], page['height']]
                    page_sizes.append(size)
                    boxes = []
                    mytxt = []

                    for bid, block in enumerate(page['blocks']):
                        bbox = block['boundingBox']

                        paragraphs = block['paragraphs']
                        rPara = ''
                        for paid, para in enumerate(paragraphs):
                            bbox = para['boundingBox']
                            rSent = ''
                            for wid, word in enumerate(para['words']):
                                symbols = word['symbols']
                                rWord = ''
                                for sid, symbol in enumerate(symbols):
                                    txt = (symbol['text'].encode("utf-8")).decode("utf-8")
                                    rWord += txt
                                rSent += ' ' + rWord
                            rPara += '' + rSent
                            mytxt.append(rSent)
                            boxes.append(getBox(bbox, size))
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
            pdf_name = content_id  # os.path.split(local_path_to_pdf)[1][:-4]
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
            gcs_source_uri = upload_blob(bucket_name, local_path_to_pdf, pdf_name + ".pdf")
            if gcs_source_uri:
                # perform GoogleOCR:
                gcs_destination_uri = "gs://{0}/{1}".format(bucket_name, os.path.split(gcs_source_uri)[1][:-4] + "/")
                print(gcs_destination_uri)
                prefix, all_text = do_GoogleOCR(gcs_source_uri, gcs_destination_uri)
                path_to_gocr_text = os.path.join(textbook_model_path, "extract", "GOCR", "text")
                path_to_gocr_json = os.path.join(textbook_model_path, "raw_data")
                if not os.path.exists(path_to_gocr_text):
                    os.makedirs(path_to_gocr_text)

                with open(os.path.join(path_to_gocr_text, prefix + ".txt"), "w") as text_file:
                    text_file.write(all_text)
                # concatenate multiple text file if any:
                textnames = findFiles(path_to_gocr_text, ["txt"])
                with open(os.path.join(path_to_gocr_text, "fulltext_annotation" + ".txt"), 'w') as outfile:
                    for fname in textnames:
                        with open(fname) as infile:
                            for line in infile:
                                outfile.write(line)
                            os.remove(fname)
                path_to_outputjson_folder = download_outputjson_reponses(bucket_name, prefix + "/",
                                                                         path_to_gocr_json, delimiter="/")
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
                    arr.append({"id": content_id + "_blob_gocr", "path": i, "Type": "gocr"})

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
    payload = "{\r\n    \"request\": {\r\n        \"filters\":{\r\n            \"identifier\":[\"" + content_id + "\"]\r\n         },\r\n               \"limit\":1\r\n    }\r\n}"
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
        path_to_toc = os.path.join(path_to_saved_folder, content_id + ".json")
        with open(path_to_toc, "w") as write_file:
            json.dump(response["result"]["content"][0], write_file, indent=4)
    except:
        pass
    return path_to_toc


def getDTB(loc):
    with open(loc) as json_file:
        DTB = json.load(json_file)
    DTB_df = []
    for topic in DTB['alignment']:
        full_text = topic["target"]['fulltext_annotation']
        cid = topic["source"]["id"]
        topic_name = topic["source"]["fulltext_annotation"]
        DTB_df.append({"identifier": cid, "name": topic_name, "text": full_text})
    return pd.DataFrame(DTB_df)


def getSimilarTopic(x, k):
    df = dist.similarity_df(x)
    similar_topic = dict()
    for i in range(len(df)):
        row_df = pd.DataFrame(df.iloc[i])
        row_df = row_df.sort_values(by=list(row_df.columns), ascending=False)
        topn = []
        for j in range(k):
            try:
                topn.append({list(row_df.index)[j]: row_df.iloc[j, 0]})
            except:
                pass
        similar_topic[list(df.index)[i]] = topn
    return similar_topic


def read_google_sheet(credentials, spreadsheet_key, worksheetpage):
    scope = ['https://spreadsheets.google.com/feeds']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials, scope)
    gc = gspread.authorize(credentials)
    spreadsheet_key = spreadsheet_key
    book = gc.open_by_key(spreadsheet_key)
    worksheet = book.worksheet(worksheetpage)
    table = worksheet.get_all_values()
    df = pd.DataFrame(table[1:], columns=table[0])
    return df


def calc_stat(list1, list2, measure):
    ls = []
    for i in range(len(list1)):
        if measure == 'division':
            try:
                ls.append(float((len(list1[i])) / (len(list2[i]))))
            except:
                ls.append(0)
        if measure == 'MED':
            try:
                ls.append(dist.getWordlistEMD((list(list1[i])), list(list2[i]), "MED"))
            except:
                ls.append(0)
    return ls


def find_span_sentence(text, sentence):
    start_index = text.find(sentence)
    end_index = start_index + len(sentence)
    return start_index, end_index


def agg_actual_predict_df(toc_df, dtb_actual, pred_df, level):
    ls = []
    for i in range(len(toc_df)):
        a = [toc_df.index[i]] + list(toc_df['Topic Name'][i][1:])
        if level == 'Topic Name':
            for j in range(len(a)):
                actual_text = " ".join(dtb_actual.loc[dtb_actual['Toc feature'] == str(a[j])]['CONTENTS'])
                try:
                    pred_text = pred_df[pred_df['title'] == a[j]]['pred_text'].iloc[0]
                except:
                    pred_text = 'nan'
                consolidated_df = pd.DataFrame([toc_df.index[i], a[j], actual_text, pred_text]).T
                consolidated_df.columns = ['ChapterName', 'TopicName', 'ActualText', 'PredictedText']
                ls.append(consolidated_df)
        elif level == 'Chapter Name':
            actual_text = " ".join(dtb_actual.loc[dtb_actual['Toc feature'].isin(a)]['CONTENTS'])
            try:
                pred_text = pred_df[pred_df['title'] == toc_df.index[i]]['pred_text'].iloc[0]
            except:
                pred_text = 'nan'
            consolidated_df = pd.DataFrame([toc_df.index[i], actual_text, pred_text]).T
            consolidated_df.columns = ['ChapterName', 'ActualText', 'PredictedText']
            ls.append(consolidated_df)
    return ls


def train_word2vec(documents, embedding_dim):
    """
    train word2vector over traning documents
    Args:
        documents (list): list of document
        embedding_dim (int): outpu wordvector size
    Returns:
        word_vectors(dict): dict containing words and their respective vectors
    """
    model = Word2Vec(documents, min_count=1, size=embedding_dim)
    word_vectors = model.wv
    del model
    return word_vectors


def get_bert_word_embeddings(documents, maxlen, bert_layer, path_to_DS_DATA_HOME):
    from tqdm import tqdm
    vect_ls = []
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    token_ls = []
    starttime = time.time()
    # take away the duplicate sentences:
    documents = list(dict.fromkeys(documents))
    for sentence in tqdm(documents):
        tokens = bert_tokenizer.tokenize(sentence)
        print("tokens: ", tokens)

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < maxlen:
            tokens = tokens + ['[PAD]' for _ in range(maxlen - len(tokens))]  # Padding sentences
        else:
            tokens = tokens[:maxlen - 1] + ['[SEP]']  # Prunning the list to be of specified max length
        token_ls.append(tokens)
        tokens_ids = bert_tokenizer.convert_tokens_to_ids(
            tokens)  # Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids)  # Converting the list to a pytorch tensor
        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_masks = (tokens_ids_tensor != 0).long()
        # bert layer:
        cont_reps, _ = bert_layer(tokens_ids_tensor.unsqueeze(0), attention_mask=attn_masks.unsqueeze(0))
        vect_dict = {}
        for i, j in enumerate(tokens):
            if j in ["[CLS]", "[SEP]", "[PAD]"]:
                pass
            else:
                vect_dict.update({j: cont_reps[:, i].detach().numpy()})
        # vect_ls.append(cont_reps.detach().numpy())
        vect_ls.append(vect_dict)
    with open(os.path.join(path_to_DS_DATA_HOME, 'embeddings_list.pkl'), 'wb') as fp:
        pickle.dump(vect_ls, fp)
    full_rep_dict = dict(ChainMap(*vect_ls))
    logging.info("BOS_PYTORCH_BERT_EMBEDDING_TIME_TAKEN: {0}mins".format((time.time() - starttime) / 60.0))
    logging.info("STOP_BOS_PYTORCH_BERT_EMBEDDING")
    return full_rep_dict


class Configuration(object):
    """Dump stuff here"""


def create_embedding_matrix(tokenizer, word_vectors, embedding_dim):
    """
    Create embedding matrix containing word indexes and respective vectors from word vectors
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object containing word indexes
        word_vectors (dict): dict containing word and their respective vectors
        embedding_dim (int): dimention of word vector
    Returns:
    """
    logging.info("START_BOS_CREATE_EMBEDDING_MATRIX")
    nb_words = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index
    print("***nb_words: ", nb_words)
    print("****embedding_dim: ", embedding_dim)
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    print("Embedding matrix shape: %s" % str(embedding_matrix.shape))
    for word, i in word_index.items():
        try:
            embedding_vector = word_vectors[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        except KeyError:
            print("vector not found for word - %s" % word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix
    logging.info("STOP_BOS_CREATE_EMBEDDING_MATRIX")


def word_embed_meta_data(documents, embedding_dim, lemmatisation, stemming):
    """
    Load tokenizer object for given vocabs list
    Args:
        documents (list): list of document
    Returns:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        embedding_matrix (dict): dict with word_index and vector mapping
    """
    documents = [x.lower().split() for x in documents]
    wordnet_lemmatizer = WordNetLemmatizer()
    porter = PorterStemmer()
    preprocess_documents = []
    if lemmatisation == True:
        for j in range(len(documents)):
            preprocess_documents.append([wordnet_lemmatizer.lemmatize(i, pos="v") for i in documents[j]])
    else:
        preprocess_documents = preprocess_documents.copy()
    preprocess_documents = []
    if stemming == True:
        for j in range(len(documents)):
            preprocess_documents.append([porter.stem(i) for i in documents[j]])
    else:
        preprocess_documents = documents.copy()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts((preprocess_documents))
    word_vector = train_word2vec(preprocess_documents, embedding_dim)
    embedding_matrix = create_embedding_matrix(tokenizer, word_vector, embedding_dim)
    del word_vector
    return tokenizer, embedding_matrix


def pytorchbert_word_embed_meta_data(documents, maxlen, bert_layer, embedding_dim, bert_tokenizer,
                                     path_to_DS_DATA_HOME):
    """
    Load tokenizer object for given vocabs list
    Args:
        documents (list): list of document
        embedding_dim (int): embedding dimension
    Returns:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        embedding_matrix (dict): dict with word_index and vector mapping
    """
    # there would have been a step to split the list of sentences to list of tokens in each sentence
    tokenizer = Tokenizer()
    # documents_ = [bert_tokenizer.tokenize(sentence) for sentence in documents]
    # tokenizer.fit_on_texts(documents)
    tokenizer.fit_on_texts(documents)
    word_vector = get_bert_word_embeddings(documents, maxlen, bert_layer, path_to_DS_DATA_HOME)
    # with open(path_to_save_embeddings, "wb") as json_file:
    #     json.dump(word_vector, json_file)
    embedding_matrix = create_embedding_matrix(tokenizer, word_vector, embedding_dim)
    del word_vector
    gc.collect()
    return tokenizer, embedding_matrix


def create_train_dev_set(tokenizer, sentences_pair, is_similar, max_sequence_length, validation_split_ratio):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        sentences_pair (list): list of tuple of sentences pairs
        is_similar (list): list containing labels if respective sentences in sentence1 and sentence2
                           are same or not (1 if same else 0)
        max_sequence_length (int): max sequence length of sentences to apply padding
        validation_split_ratio (float): contain ratio to split training data into validation data
    Returns:
        train_data_1 (list): list of input features for training set from sentences1
        train_data_2 (list): list of input features for training set from sentences2
        labels_train (np.array): array containing similarity score for training data
        leaks_train(np.array): array of training leaks features
        val_data_1 (list): list of input features for validation set from sentences1
        val_data_2 (list): list of input features for validation set from sentences1
        labels_val (np.array): array containing similarity score for validation data
        leaks_val (np.array): array of validation leaks features
    """
    start = time.time()
    logging.info("START_BOS_CREATE_TRAIN")
    sentences1 = [x[0].lower() for x in sentences_pair]
    sentences2 = [x[1].lower() for x in sentences_pair]
    train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sequences_2 = tokenizer.texts_to_sequences(sentences2)
    leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(train_sequences_1, train_sequences_2)]

    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=max_sequence_length)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=max_sequence_length)
    train_labels = np.array(is_similar)
    leaks = np.array(leaks)

    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]
    leaks_shuffled = leaks[shuffle_indices]

    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))

    del train_padded_data_1
    del train_padded_data_2
    gc.collect()

    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]
    leaks_train, leaks_val = leaks_shuffled[:-dev_idx], leaks_shuffled[-dev_idx:]
    logging.info("STOP_BOS_CREATE_TRAIN")
    logging.info("BOS_CREATE_TRAIN_TIME_TAKEN: {0}mins".format((time.time() - start) / 60.0))
    return train_data_1, train_data_2, labels_train, leaks_train, val_data_1, val_data_2, labels_val, leaks_val


def generate_tokenizer_embedding_mat(embedding_args, embedding_algo, complete_df, path_to_DS_DATA_HOME):
    embedding_dim = int(embedding_args[embedding_algo]["EMBEDDING_DIM"])
    print("*****embed_dim: ", embedding_dim)
    print("*****path to DS_DATA_HOME: ", path_to_DS_DATA_HOME)
    if embedding_algo == "BERT":
        maxlen = int(embedding_args[embedding_algo]["max_seq_length"])
        bert_layer = BertModel.from_pretrained('bert-base-uncased')
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        tokenizer, embedding_matrix = pytorchbert_word_embed_meta_data(
            list(complete_df['sentence1']) + list(complete_df['sentence2']), maxlen, bert_layer, embedding_dim,
            bert_tokenizer, path_to_DS_DATA_HOME)

    if embedding_algo == "Word2Vec":
        lemmatisation = embedding_args[embedding_algo]["lemmatisation"]
        stemming = embedding_args[embedding_algo]["stemming"]
        tokenizer, embedding_matrix = word_embed_meta_data(
            list(complete_df['sentence1']) + list(complete_df['sentence2']), embedding_dim, lemmatisation, stemming)

    embedding_meta_data = {
        'tokenizer': tokenizer,
        'embedding_matrix': embedding_matrix
    }
    return embedding_meta_data


def create_test_data(tokenizer, test_sentences_pair, max_sequence_length):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        test_sentences_pair (list): list of tuple of sentences pairs
        max_sequence_length (int): max sequence length of sentences to apply padding
    Returns:
        test_data_1 (list): list of input features for training set from sentences1
        test_data_2 (list): list of input features for training set from sentences2
    """
    start = time.time()
    logging.info("START_BOS_CREATE_TEST")
    test_sentences1 = [x[0].lower() for x in test_sentences_pair]
    test_sentences2 = [x[1].lower() for x in test_sentences_pair]

    test_sequences_1 = tokenizer.texts_to_sequences(test_sentences1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_sentences2)
    leaks_test = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
                  for x1, x2 in zip(test_sequences_1, test_sequences_2)]

    leaks_test = np.array(leaks_test)
    test_data_1 = pad_sequences(test_sequences_1, maxlen=max_sequence_length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=max_sequence_length)
    logging.info("STOP_BOS_CREATE_TEST")
    logging.info("BOS_CREATE_TEST_TIME_TAKEN: {0}mins".format((time.time() - start) / 60.0))

    return test_data_1, test_data_2, leaks_test


class SiameseBiLSTM:
    def __init__(self, embedding_dim, max_sequence_length, number_lstm, number_dense, rate_drop_lstm,
                 rate_drop_dense, hidden_activation, validation_split_ratio):
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.number_lstm_units = number_lstm
        self.rate_drop_lstm = rate_drop_lstm
        self.number_dense_units = number_dense
        self.activation_function = hidden_activation
        self.rate_drop_dense = rate_drop_dense
        self.validation_split_ratio = validation_split_ratio

    def train_model(self, sentences_pair, is_similar, embedding_meta_data, model_save_directory):
        """
        Train Siamese network to find similarity between sentences in `sentences_pair`
            Steps Involved:
                1. Pass the each from sentences_pairs  to bidirectional LSTM encoder.
                2. Merge the vectors from LSTM encodes and passed to dense layer.
                3. Pass the  dense layer vectors to sigmoid output layer.
                4. Use cross entropy loss to train weights
        Args:
            sentences_pair (list): list of tuple of sentence pairs
            is_similar (list): target value 1 if same sentences pair are similar otherwise 0
            embedding_meta_data (dict): dict containing tokenizer and word embedding matrix
            model_save_directory (str): working directory for where to save models
        Returns:
            return (best_model_path):  path of best model
        """
        start = time.time()
        logging.info("START_BOS_TRAIN_MODEL")
        tokenizer, embedding_matrix = embedding_meta_data['tokenizer'], embedding_meta_data['embedding_matrix']

        train_data_x1, train_data_x2, train_labels, leaks_train, \
        val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(tokenizer, sentences_pair,
                                                                               is_similar, self.max_sequence_length,
                                                                               self.validation_split_ratio)
        if train_data_x1 is None:
            print("++++ !! Failure: Unable to train model ++++")
            return None

        nb_words = len(tokenizer.word_index) + 1

        # Creating word embedding layer
        embedding_layer = Embedding(nb_words, self.embedding_dim, weights=[embedding_matrix],
                                    input_length=self.max_sequence_length, trainable=False)

        # Creating LSTM Encoder--> try GRU
        lstm_layer = Bidirectional(
            LSTM(self.number_lstm_units, dropout=self.rate_drop_lstm, recurrent_dropout=self.rate_drop_lstm))

        # Creating LSTM Encoder layer for First Sentence
        sequence_1_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        x1 = lstm_layer(embedded_sequences_1)

        # Creating LSTM Encoder layer for Second Sentence
        sequence_2_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        x2 = lstm_layer(embedded_sequences_2)

        # Creating leaks input
        leaks_input = Input(shape=(leaks_train.shape[1],))
        leaks_dense = Dense(int(self.number_dense_units / 2), activation=self.activation_function)(leaks_input)

        # Merging two LSTM encodes vectors from sentences to
        # pass it to dense layer applying dropout and batch normalisation
        merged = concatenate([x1, x2, leaks_dense])
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        merged = Dense(self.number_dense_units, activation=self.activation_function)(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        preds = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[sequence_1_input, sequence_2_input, leaks_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        STAMP = 'lstm_%d_%d_%.2f_%.2f' % (self.number_lstm_units, self.number_dense_units,
                                          self.rate_drop_lstm, self.rate_drop_dense)

        checkpoint_dir = model_save_directory + '/' + 'checkpoints/' + str(int(time.time())) + '/'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        bst_model_path = checkpoint_dir + STAMP + '.h5'

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

        tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

        model_ = model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
                           validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels),
                           epochs=200, batch_size=64, shuffle=True,
                           callbacks=[early_stopping, model_checkpoint, tensorboard],
                           )
        model_summary = json.loads(str(model.to_json()))
        with open(os.path.join(model_save_directory, 'model_summary.json'), "w") as summ_file:
            json.dump(model_summary, summ_file, indent=4)
        # with open(os.path.join(path_to_DS_DATA_HOME, 'model_history.json'), "w") as hist_file:
        #     json.dump(model_.history, hist_file, indent=4)
        loss_history = model_.history["loss"]
        numpy_loss_history = np.array(loss_history)
        df = pd.DataFrame(numpy_loss_history, columns=["loss"])
        df['val_loss'] = np.array(model_.history["val_loss"])
        df['accuracy'] = np.array(model_.history["acc"])
        df.to_csv(os.path.join(model_save_directory, "loss_function_df.csv"))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(df.index), y=list(df['loss']),
                                 mode='lines',
                                 name='train'))
        fig.add_trace(go.Scatter(x=list(df.index), y=list(df['val_loss']),
                                 mode='lines',
                                 name='test'))
        fig.update_layout(title='model loss',
                          xaxis_title='epoch',
                          yaxis_title='loss')
        py.offline.plot(fig, filename=os.path.join(model_save_directory, 'loss_function_curve.html'))
        logging.info("STOP_BOS_TRAIN_MODEL")
        logging.info("BOS_CREATE_TRAIN_MODEL_TIME_TAKEN: {0}mins".format((time.time() - start) / 60.0))
        return bst_model_path

    def update_model(self, saved_model_path, new_sentences_pair, is_similar, embedding_meta_data):
        """
        Update trained siamese model for given new sentences pairs 
            Steps Involved:
                1. Pass the each from sentences from new_sentences_pair to bidirectional LSTM encoder.
                2. Merge the vectors from LSTM encodes and passed to dense layer.
                3. Pass the  dense layer vectors to sigmoid output layer.
                4. Use cross entropy loss to train weights
        Args:
            model_path (str): model path of already trained siamese model
            new_sentences_pair (list): list of tuple of new sentences pairs
            is_similar (list): target value 1 if same sentences pair are similar otherwise 0
            embedding_meta_data (dict): dict containing tokenizer and word embedding matrix
        Returns:
            return (best_model_path):  path of best model
        """
        start = time.time()
        logging.info("START_BOS_UPDATE_MODEL")
        tokenizer = embedding_meta_data['tokenizer']
        train_data_x1, train_data_x2, train_labels, leaks_train, \
        val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(tokenizer, new_sentences_pair,
                                                                               is_similar, self.max_sequence_length,
                                                                               self.validation_split_ratio)
        model = load_model(saved_model_path)
        model_file_name = saved_model_path.split('/')[-1]
        new_model_checkpoint_path = saved_model_path.split('/')[:-2] + str(int(time.time())) + '/'

        new_model_path = new_model_checkpoint_path + model_file_name
        model_checkpoint = ModelCheckpoint(new_model_checkpoint_path + model_file_name,
                                           save_best_only=True, save_weights_only=False)

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        tensorboard = TensorBoard(log_dir=new_model_checkpoint_path + "logs/{}".format(time.time()))

        model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
                  validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels),
                  epochs=50, batch_size=3, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])
        logging.info("STOP_BOS_UPDATE_MODEL")
        logging.info("BOS_UPDATE_MODEL_TIME_TAKEN: {0}mins".format((time.time() - start) / 60.0))
        return new_model_path


def generate_pr_curve(predicted_model, path_to_DS_DATA_HOME):
    precision, recall, threshold = precision_recall_curve(predicted_model['actual_label'].ravel(),
                                                          predicted_model['pred_score'].ravel())
    average_precision = average_precision_score(predicted_model['actual_label'].ravel(),
                                                predicted_model['pred_score'].ravel())
    trace1 = go.Scatter(x=recall, y=precision,
                        mode='lines',
                        hovertext=threshold,
                        line=dict(width=2, color='navy'),
                        name='Precision-Recall curve')

    layout = go.Layout(title='Precision-Recall Curve: AUC={0:0.2f}'.format(average_precision),
                       xaxis=dict(title='Recall'),
                       yaxis=dict(title='Precision'))

    fig = go.Figure(data=[trace1], layout=layout)
    py.offline.plot(fig, filename=os.path.join(path_to_DS_DATA_HOME, 'pr_curve.html'))
    # fig.write_image(path_to_save_image)
    # return path_to_save_image


def filter_by_grade_range(df, grade_range):
    df_list = []
    for grade in df['STB_Grade'].unique():
        temp_df = df[df['STB_Grade'] == grade]
        grade = int(grade.split()[-1])
        x = ['Class ' + str(i) for i in range(grade - grade_range, grade + (grade_range + 1))]
        temp_df = temp_df[temp_df['ref_grade'].isin(x)]
        df_list.append(temp_df)
    final_df = pd.concat(df_list)
    return final_df


def create_scoring_data(tokenizer, score_sentences_pair, max_sequence_length):
    """
    Create scoring dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        test_sentences_pair (list): list of tuple of sentences pairs
        max_sequence_length (int): max sequence length of sentences to apply padding
    Returns:
        test_data_1 (list): list of input features for training set from sentences1
        test_data_2 (list): list of input features for training set from sentences2
    """
    start = time.time()
    logging.info("START_BOS_CREATE_TEST")
    test_sentences1 = [x[0].lower() for x in score_sentences_pair]
    test_sentences2 = [x[1].lower() for x in score_sentences_pair]

    test_sequences_1 = tokenizer.texts_to_sequences(test_sentences1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_sentences2)
    leaks_test = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
                  for x1, x2 in zip(test_sequences_1, test_sequences_2)]

    leaks_test = np.array(leaks_test)
    test_data_1 = pad_sequences(test_sequences_1, maxlen=max_sequence_length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=max_sequence_length)
    logging.info("STOP_BOS_CREATE_TEST")
    logging.info("BOS_CREATE_TEST_TIME_TAKEN: {0}mins".format((time.time() - start) / 60.0))
    return test_data_1, test_data_2, leaks_test


def scoring_module(tokenizer, best_model_path, siamese_config, test_df, threshold):
    """
    Score a test dataset using pretrained BERT model and pickled tokenizer which is trained on the train dataset.
    Args:
        tokenizer: Tokenizer object which is trained on the train dataset
        best_model_path: Path to the pretrained BERT model
        siamese_config: The configuration of the pretrained model
        threshold: The value that is to be exceded for the label to be 1 or 0
    Returns:
        model_pred_df (dataframe): dataframe with predicted score and predicted label after applying threshold on the score.
    """
    test_sentence_pairs = [(x1, x2) for x1, x2 in zip(list(test_df['sentence1']), list(test_df['sentence2']))]
    test_data_x1, test_data_x2, leaks_test = create_scoring_data(tokenizer, test_sentence_pairs,
                                                                 siamese_config['MAX_SEQUENCE_LENGTH'])
    best_model = load_model(best_model_path, compile=False)
    preds = list(best_model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
    results_ = [(x, y, z) for (x, y), z in zip(test_sentence_pairs, preds)]
    results = []
    for i in range(len(results_)):
        results.append(tuple(list(test_df.iloc[i]) + list(results_[i])))
    results.sort(key=itemgetter(2), reverse=True)
    model_pred_df = pd.DataFrame(results)
    model_pred_df.columns = test_df.columns.to_list() + ['sentence1_score', 'sentence2_score', 'pred_score']
    model_pred_df['predicted_label'] = np.where(model_pred_df['pred_score'] > threshold, 1, 0)
    model_pred_df.drop(['sentence2_score', 'sentence2_score'], axis=1, inplace=True)
    return model_pred_df


def create_confusion_matrix(row):
    """
    Reports the number of false positives, false negatives, true positives, and true negatives
    to describe the performance of the model
    Args:
        row: A row in the input dataframe
    Returns:
        Accuracy variable(variable): Returns the accuracy measure that the predicted label corresponds in a row
    """
    if row['actual_label'] == 1:
        if row['predicted_label'] == 1:
            return 'TP'
        else:
            return 'FN'
    else:
        if row['predicted_label'] == 1:
            return 'FP'
        else:
            return 'TN'


def aggregation_topic_level(output_df, aggregation_criteria, mandatory_column_names):
    """
    Aggregate output of the sentence similarity model at topic level
    Args:
        output_df: Predicted output dataframe
        aggregation_criteria: The method of aggregation that needs to be performed
        mandatory_column_names: Dictionary of columns names for the output dataframe
    Returns:
        full_score_df(dataframe): Dataframe with aggregated similarity score at topic level for a given pair of
             state topic ID and reference topic ID.
    """
    full_df_ls = []
    if 'cm' not in output_df.columns:
        output_df['cm'] = output_df.apply(create_confusion_matrix, axis=1)
    try:
        for stb_topic_id in output_df["stb_id"].unique():
            big_ls = []
            stb_df = output_df[output_df["stb_id"] == stb_topic_id]
            for ref_id in stb_df["ref_id"].unique():
                pred_score_percent = 0
                stb_1_df_sam = stb_df[stb_df["ref_id"] == ref_id]
                eval_dict = dict(stb_1_df_sam['cm'].value_counts())
                print("eval_dict: ", eval_dict)
                columns = [v for k, v in mandatory_column_names.items()]
                score_df = pd.DataFrame(index=[0], columns=columns)
                score_df = score_df.fillna(0)
                if aggregation_criteria == "average":
                    pred_score_percent = (stb_1_df_sam['predicted_label'].sum() / len(stb_1_df_sam)) * 100
                score_df[mandatory_column_names["stb_topic_col_name"]] = stb_topic_id
                score_df[mandatory_column_names["ref_topic_col_name"]] = ref_id
                score_df[mandatory_column_names["pred_agg_col_name"]] = pred_score_percent
                score_df[mandatory_column_names["label_col_name"]] = stb_1_df_sam['actual_label'].mean()
                if "FP" in eval_dict.keys():
                    score_df[mandatory_column_names["fp_col_name"]] = eval_dict["FP"]
                if "TN" in eval_dict.keys():
                    score_df[mandatory_column_names["tn_col_name"]] = eval_dict["TN"]
                if "TP" in eval_dict.keys():
                    score_df[mandatory_column_names["tp_col_name"]] = eval_dict["TP"]
                if "FN" in eval_dict.keys():
                    score_df[mandatory_column_names["fn_col_name"]] = eval_dict["FN"]
                big_ls.append(score_df)
            full_score_df = pd.concat(big_ls).reset_index(drop=True).sort_values(
                by=[mandatory_column_names["pred_agg_col_name"]], ascending=False)
            full_df_ls.append(full_score_df)
        full_score_df = pd.concat(full_df_ls).reset_index(drop=True)
        return full_score_df
    except:
        print("Error occurred!!")


def k_topic_recommendation(full_score_df, window):
    """
    Generates a dataframe with  top n reference topics ID for a given state topic ID

    Args:
        full_score_df: Dataframe with aggregated similarity score at topic level for a given pair of
            state topic ID and reference topic ID.
        window: The number of recommendations to be displayed
    Returns:
        big_full_score(dataframe): Dataframe with recommendations from 1 to window size.
    """
    full_df_ls = []
    df_ls = []
    for stb_topic_id in full_score_df["stb_id"].unique():
        stb_df = full_score_df[full_score_df["stb_id"] == stb_topic_id]
        actual_label_max_score = max(stb_df["actual_label"].unique())
        actual_ref_id = stb_df[stb_df["actual_label"] == actual_label_max_score]["ref_id"].iloc[0]
        print("actual ref id: ", actual_ref_id)
        grouped_refid_df_ = stb_df.groupby('pred_label_percentage')['ref_id'].apply(list).reset_index(
            name='grouped_ref_id').sort_values(by=['pred_label_percentage'], ascending=False)[:window]
        grouped_refid_df_["stb_id"] = stb_topic_id
        full_df_ls.append(grouped_refid_df_)
        columns = ["stb_id", "k=1", "k=2", "k=3", "k=4", "k=5"]
        df_ = pd.DataFrame(index=[0], columns=columns)
        df_ = df_.fillna(0)
        df_["stb_id"] = stb_topic_id
        for i in range(window):
            if actual_ref_id in grouped_refid_df_["grouped_ref_id"].iloc[i]:
                df_.loc[0, i + 1:] = 1
                break
        df_ls.append(df_)
    big_full_score = pd.concat(df_ls).reset_index(drop=True)
    # grouped_refid_df = pd.concat(full_df_ls).reset_index(drop=True)
    return big_full_score


def listify(x):
    """
    x: Pandas series
    return: list object of string column
    """
    try:
        return ast.literal_eval(x)
    except ValueError:
        return None


def modify_df(df, sentence_length):
    """
    Convert the columns STB_Text from a string of list to a list.
    Gather length of STB_Text/Ref_Text list and filter based on sentence length argument.
    Explode STB_Text and Ref_Text as separate dataframes.
    Filter repetitive sentences within a topic.
    Create a column to join on and a column for unique topic-sentence id.
    :param df: base data frame which has ['STB_Id', 'STB_Grade', 'STB_Section', 'STB_Text', 'Ref_id', 'Ref_Grade',
        'Ref_Section', 'Ref_Text'] columns
    :param sentence_length:  drop row if topic has number of sentences less than or equal to sentence_length
    :return: exploded, enriched STB and Ref data frames
    """
    df['STB_Text'] = df['STB_Text'].apply(listify)
    df['Ref_Text'] = df['Ref_Text'].apply(listify)
    df.dropna(axis=0, subset=['STB_Text', 'Ref_Text'], inplace=True)
    df['STB_Text_len'] = df['STB_Text'].apply(lambda x: len(x))
    df['Ref_Text_len'] = df['Ref_Text'].apply(lambda x: len(x))
    df = df[df['STB_Text_len'] > sentence_length]
    df = df[df['Ref_Text_len'] > sentence_length]
    df.drop(['STB_Text_len', 'Ref_Text_len'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    stb_df = df[['STB_Id', 'STB_Grade', 'STB_Section', 'STB_Text', 'Ref_id']]
    stb_df = stb_df.explode('STB_Text')
    stb_df['join_id'] = 1
    stb_df.drop_duplicates(subset=['STB_Id', 'STB_Text'], inplace=True)
    stb_df.reset_index(drop=True, inplace=True)
    stb_df.reset_index(inplace=True)
    ref_df = df[['Ref_id', 'Ref_Grade', 'Ref_Section', 'Ref_Text']]
    ref_df = ref_df.explode('Ref_Text')
    ref_df['join_id'] = 1
    ref_df.drop_duplicates(subset=['Ref_id', 'Ref_Text'], inplace=True)
    ref_df.reset_index(drop=True, inplace=True)
    ref_df.reset_index(inplace=True)
    return stb_df, ref_df


def generate_cosine_similarity_score(stb_df, ref_df, input_folder_path):
    """
    Given corpus and limiter to differentiate stb sentences from ref sentences, generate cosine similarity score for
    each sentence pair across stb and ref df
    :param stb_df: dataframe consisting of stb sentences
    :param ref_df: dataframe consisting of ref sentences
    :param input_folder_path: folder path where to save the cosine similarity matrix
    :return: cosine similarity matrix for the sentence pairs
    """
    corpus = stb_df['STB_Text'].tolist() + ref_df['Ref_Text'].tolist()
    limiter = stb_df.shape[0]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), analyzer='word')
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()
    similarity = cosine_similarity(X[:limiter], X[limiter:])
    with open(os.path.join(input_folder_path, 'cosine_similarity.pkl'), 'wb') as f:
        pickle.dump(similarity, f)
    cos_sim = pd.DataFrame(pd.DataFrame(similarity).stack())
    cos_sim.index.names = ['index_stb', 'index']
    cos_sim.columns = ['cos_sim_score']
    return cos_sim


def append_cosine_similarity_score(stb_df, ref_df, cos_sim, input_folder_path):
    """
    join stb, ref and cosine similarity dataframes together
    :param stb_df: data frame consisting of ['STB_Id', 'STB_Grade', 'STB_Section', 'STB_Text', 'Ref_id'] columns
    :param ref_df: data frame consisting of ['Ref_id', 'Ref_Grade', 'Ref_Section', 'Ref_Text'] columns
    :param cos_sim: data frame consisting of ['cos_sim_score'] column
    :param input_folder_path: folder path where to save the complete data set
    :return:
    """
    jdf = stb_df.set_index('join_id').join(ref_df.set_index('join_id'), how='left', lsuffix='_stb')
    jdf.set_index(['index_stb', 'index'], inplace=True)
    jdf = jdf.join(cos_sim, how='left')
    jdf.index.names = ['stb_sent_id', 'ref_sent_id']
    jdf['actual_label'] = jdf.apply(lambda x: 1 if x['Ref_id_stb'] == x['Ref_id'] else 0, axis=1)
    jdf.drop('Ref_id_stb', axis=1, inplace=True)
    jdf.columns = ['stb_id', 'stb_grade', 'stb_topic', 'sentence1', 'ref_id', 'ref_grade', 'ref_topic', 'sentence2',
                   'cos_sim_score', 'actual_label']
    jdf.to_csv(os.path.join(input_folder_path, 'complete_data_set.csv'))
