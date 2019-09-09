import os
import daggit
import requests
import io
import re
import shutil
import json
import numpy as np
import pandas as pd
import Levenshtein
from pyemd import emd

import copy
import json


def filterText(text, apply=True):

    # get rid of newlines
    if apply:
        text = text.strip().replace("\n", " ").replace("\r","")
        #text = text.lower()
        # remove spaces, periods brackets
        text = re.sub(r'\.(?!\d)|\s+', '', text)
        # remove [] brackets
        text = re.sub(r'\([^)]*\)', '', text)
        # remove () brackets
        text = re.sub(r'\(|\)|\[|\]', '' ,text)
    return text

def readSentences(fname,filter=True):
    lines = {}
    counter = 0
    with open(fname) as f:
        for i,line in enumerate(f):
            text = filterText(line,apply=filter)
            if text:
                lines[i] = text
    return lines


def minedit(a,b,filter=True):
    a = filterText(a, apply=filter)
    b = filterText(b, apply=filter)
    d = 1.0*Levenshtein.distance(a, b)/max(len(a),len(b))
    return d

def getPairwiseDist(row_strings,col_strings,dist_method='minedit'):
    # Levenshtein distance
    if dist_method == 'minedit':
        f = minedit
    else:
        f = minedit
    n = len(row_strings)
    m = len(col_strings)
    dist_matrix= np.zeros((n,m))

    for i,row_word in enumerate(row_strings):
        for j,col_word in enumerate(col_strings):
            dist_matrix[i,j] = f(row_word,col_word)
    return dist_matrix

def getDocDistance(doc_a, doc_b, pooling = 'avg', dist_method='minedit'):
    dist = getPairwiseDist(doc_a.split(),doc_b.split(),dist_method=dist_method)
    d = dist.min()
    return d

def create_reverse_lookup(doc):
    k = len(doc)
    doc_reverse_map = {}
    for i,line in enumerate(doc.items()):
        doc_reverse_map[i]=line[0]
    return doc_reverse_map

def getTocToDocDist(toc,doc,dist_method='minedit'):
    # Levenshtein distance
    if dist_method == 'minedit':
        f = minedit
    else:
        f = minedit
    k = len(toc)
    m = len(doc)
    w = np.zeros((k,m))
    for i,title in enumerate(toc.items()):
        for j,sent in enumerate(doc.items()):
            w[i,j] = minedit(title[1],sent[1])
    return w

def getBreakPoints(w,cutoff=0.0001,toc=True):
    k,m=w.shape
    bkps = []
    for ind in range(k):
        signal = w[ind,].ravel()
        index = np.argsort(signal)[:2]
        index = np.sort(index)
        if toc:
            bkps.append(index[1])
    bkps.append(m)
    return bkps

def readToC(f_toc,col_name='Chapter Name',filter=True):
    df = pd.read_csv(f_toc)
    x = df.drop_duplicates(subset=col_name)
    toc = x[col_name].to_dict()
    toc_id = list(x["Identifier"])
    for ind,val in toc.items():
        text = filterText(val,apply=filter)
        toc[ind] = text
    return toc, toc_id


def create_dtb(f_toc, f_text):


    toc,toc_id = readToC(f_toc,filter=False)
    doc = readSentences(f_text,filter=False)

    doc_reverse_map = create_reverse_lookup(doc)
    toc_reverse_map = create_reverse_lookup(toc)
    w = getTocToDocDist(toc,doc)
    bkps = getBreakPoints(w)

    for tp, sp in enumerate(bkps[:-1]):
        doc_index = doc_reverse_map[sp]
        toc_index = toc_reverse_map[tp]
        print('toc',toc[toc_index],'span',doc_index,'doc',doc[doc_index],' > ',)
        print('\n')

    dtb = {}
    dtb_blob_array = []

    dtb_blob = {'id':None,'span':{'start':0,'end':0,'atomicity':'line'},'path':None,'fulltext_annotation':None}

    for tp, sp in enumerate(bkps[:-1]):

        doc_start_index = doc_reverse_map[sp]
        toc_start_index = toc_reverse_map[tp]

        doc_end_index = doc_reverse_map[bkps[tp+1]-1]
        toc_end_index = toc_start_index

        toc_blob = copy.deepcopy(dtb_blob)
        doc_blob = copy.deepcopy(dtb_blob)

        toc_blob['id'] = toc_id[tp]
        toc_blob['span']['start'] = toc_start_index
        toc_blob['span']['end'] = toc_end_index

        doc_blob['span']['start'] = doc_start_index
        doc_blob['span']['end'] = doc_end_index

        toc_blob['fulltext_annotation'] = toc[toc_start_index]
        doc_blob['fulltext_annotation'] = ''.join([doc[ind] for ind in range(doc_start_index,doc_end_index)])

        dtb_blob_array.append({'source':toc_blob,'target':doc_blob})


    dtb['alignment']=dtb_blob_array
    return dtb
