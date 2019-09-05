import re
import string
import pandas as pd
import numpy as np
from pyemd import emd
import Levenshtein
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import gensim.downloader as api
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from bert_serving.client import BertClient
from daggit.core.oplib import nlp as preprocess

def getMinEditDist(l1,l2):
    # Levenshtein distance
    dist_matrix=[]
    for row_word in l1:
        row_vec=[]
        for col_word in l2:
            row_vec.append(1.0*Levenshtein.distance(row_word, col_word)/max(len(row_word),len(col_word)))
        dist_matrix.append(row_vec)
    dist=(pd.DataFrame(dist_matrix,columns=l2, index=l1))
    return dist

def getpairwiseDist(l1,l2):
    # based on pairwise allignment of words- biopytgon
    dist_matrix=[]
    for row_word in l1:
        row_vec=[]
        for col_word in l2:
            alignments = pairwise2.align.globalxx(row_word.decode('utf-8'), col_word.decode('utf-8'))
            row_vec.append(1-(1.0*alignments[0][2]/alignments[0][4]))
        dist_matrix.append(row_vec)
    dist=(pd.DataFrame(dist_matrix,columns=l2, index=l1))
    return dist

def similarity_df(x):
    sim = 1.0/(1+x)
    return sim


def getWordlistEMD(doc1, doc2, method):
    """
    A custom function to find distance between two documents.
    :param doc1(list of strings): A tokenised string of length `l`.
    :param doc2(list of strings): A tokenised string of length `m`.
    "param method(string): Method used for computing distance(MED or globalxx)
    :returns: emd between the two docs
    """
    keyword_union=doc1+doc2

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit(keyword_union)
    hist1=X.transform([', '.join(doc1)]).todense()
    hist2=X.transform([', '.join(doc2)]).todense()
    h1=normalize(np.array(np.array(hist1)).astype('float64'))#preserve the order
    h2=normalize(np.array(np.array(hist2)).astype('float64'))
    keyword_union_set=(X.vocabulary_).keys()
    if (method=='MED'):
        word_dist=(getMinEditDist(keyword_union_set,keyword_union_set).values).astype('float64')
    elif (method=='globalxx'):
        word_dist=(getpairwiseDist(keyword_union_set,keyword_union_set).values).astype('float64')
    else :
        print("Invalid input")

    word_dist = word_dist.copy(order='C')
    dist=emd(h1.ravel(), h2.ravel(),word_dist)
    return dist

def getBertCosine(list1, list2):
    """
    Requires bert server running.
    bert-serving-start -model_dir /uncased_L-12_H-768_A-12/ -num_worker=1

    """
    try:
        bc = BertClient()
        emb1=bc.encode(list1)
        emb2=bc.encode(list2)
        #.tolist()
        distance = pd.DataFrame(sklearn.metrics.pairwise.cosine_distances(emb1, emb2))
        return distance
    except:
        print("Requires BERT server. Unable to detect.")


def getDistance(list1, list2, method):
    """
    A custom function to find distance between two lists.
    :param list1(list of list of strings): list of tokenised strings of length `l`.
    :param list2(list of list of strings): list of tokenised string of length `m`.
    "param method(string): Method used for computing distance(EMD- using pyemd or WMD- using gensim wmd, BERT- using BERT with cosine distance)
    :returns: a distance matrix of dim l X m, similaritymatrix of dim l X m.
    """

    row_num=len(list1)
    col_num=len(list2)
    dist = pd.DataFrame(np.zeros([row_num,col_num]), index=range(0,row_num), columns=range(0,col_num) )

    if method!="BERT":
        list1_tokens = [preprocess.tokenize(i) for i in preprocess.clean_string_list(list1)]
        list2_tokens = [preprocess.tokenize(i) for i in preprocess.clean_string_list(list2)]


        if method=="WMD":
            word_vectors = api.load("glove-wiki-gigaword-100")

        for i in range(0,row_num):
            for j in range(0,col_num):
                if method=="EMD":
                    dist.iloc[i,j]=getWordlistEMD(list1_tokens[i], list2_tokens[j],"MED")
                elif method=="WMD":
                     dist.iloc[i,j]=word_vectors.wmdistance(list1_tokens[i], list2_tokens[j])
                else :
                    print("Invalid method for distance computation")

    else :
        dist = getBertCosine(list1, list2)
    return dist.values.tolist()
