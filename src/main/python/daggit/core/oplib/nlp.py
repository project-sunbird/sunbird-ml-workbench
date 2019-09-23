import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from google.cloud import translate
from nltk.corpus import stopwords

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
import json
import re
import io
import time
import requests
import os
import logging
import xml.etree.ElementTree as ET
from dateutil.parser import parse
logging.getLogger("requests").setLevel(logging.WARNING)

# nltk.download("stopwords")
# nltk.download("wordnet")
stopwords = stopwords.words('english')


pun_list = list(string.punctuation)

def language_detection(text):

    """
    This function will take in an enriched text as input and
    use google translate API to detect language of the text and returns it


    :param text(str): The text for which the language need to be detected.
    :returns: The detected language for the given text.
    """
    translate_client = translate.Client()
    result = translate_client.detect_language(text)
    return result["language"]


def clean_text(text): #cleantext
    """
    Custom function to clean enriched text

    :param text(str): Enriched ytext from Ekstep Content.
    :returns: Cleaned text.
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

def tokenize(text, tokenizer=nltk.word_tokenize): #clean_text_tokens
    """
    A custom preprocessor to tokenise and clean a text.
    Used in Content enrichment pipeline.

    Process:

    * tokenise string using nltk word_tokenize()

    * Remove stopwords

    * Remove punctuations in words

    * Remove digits and whitespaces

    * Convert all words to lowercase

    * Remove words of length

    * Remove nan or empty string

    :param text(str): The string to be tokenised.
    :returns: List of cleaned tokenised words.
    """
    tokens = tokenizer(text)
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

    :param word(str): Typically a word whose punctuations and space are removed.
    :param DELIMITTER(str): String to replace punctuations.

    :returns: Processed string.
    """
    delimitters = ["___", "__", " ", ",", "_", "-", ".", "/"] + \
        list(set(string.punctuation))
    for lim in delimitters:
        word = word.replace(lim, delimitter)
    return word


def strip_word_number(list_word, delimitter):
    delimitters = ["___", "__", " ", ",", "_", "-", ".", "/","\n"+"–","–","’","‘"] + \
        list(set(string.punctuation))
    for lim in delimitters:
        list_word = [ str(i).replace(lim, delimitter).replace("\n"," ") for i in list_word]
        list_word = [ i.lower().strip() for i in list_word ]
        list_word = [ re.sub(r'\b[0-9]+\b\s*', '', i) for i in list_word ] 
        list_word = [re.sub(' +', ' ', i) for i in list_word]
    return list_word


def get_tokens(path_to_text_file, tokenizer=nltk.word_tokenize): #custom_tokenizer
    """
    Given a text file uses custom_tokenizer function
    to tokenise and write the tokenised words to a keywords.csv file.

    :param path_to_text_file(str): Location of text file to be tokenised
    :param path_to_text_tokens_folder(str): Location to write the tokenised words

    :returns: A dataframe with a ``KEYWORDS`` column that contains tokenised keywords.
    """
    text = open(path_to_text_file, "r")
    text_file = text.read()
    text_list =tokenize(text_file, tokenizer)
    text_df = pd.DataFrame(text_list, columns=['KEYWORDS'])
    return text_df


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


def custom_listPreProc(key_list, preproc, DELIMITTER): #custom_listPreProc
    key_list = [clean_string_list(x) for x in key_list]
    key_list_clean = []
    for x in key_list:
        x = [strip_word(i, DELIMITTER) for i in x]
        key_list_clean.append(stem_lem((x), DELIMITTER))
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


def get_phrase(x_list, DELIMITTER): #getPhrase
    # x_list=clean_string_list(x_list)
    x_phrase = [i for i in x_list if DELIMITTER in i]
    x_word = [item for item in x_list if item not in x_phrase]
    return x_word, x_phrase


def remove_short_words(mylist, wordlen):#removeShortWords
    return [item for item in mylist if len(item) > wordlen]


def word_to_phrase_match(wordlist, phraselist, DELIMITTER):#WordtoPhraseMatch
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
    list1 = remove_short_words(list1, 0)
    list2 = remove_short_words(list2, 0)
    list1_words, list1_phrases = get_phrase(list1, DELIMITTER)
    list2_words, list2_phrases = get_phrase(list2, DELIMITTER)
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
    partial_match_list1, count = word_to_phrase_match(
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
    partial_match_list2, count = word_to_phrase_match(
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