import os
import re
import unicodedata
import sys

import gensim
import pandas as pd

if sys.version_info[0] >= 3:
    unicode = str


PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE)
RE_HTML_ENTITY = re.compile(r'&(#?)([xX]?)(\w{1,8});', re.UNICODE)

#for gensim operators

def to_unicode(text, encoding='utf8', errors='strict'):
    """Convert `text` to unicode.
    Parameters
    ----------
    text : str
        Input text.
    errors : str, optional
        Error handling behaviour, used as parameter for `unicode` function (python2 only).
    encoding : str, optional
        Encoding of `text` for `unicode` function (python2 only).
    Returns
    -------
    str
        Unicode version of `text`.
    """
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)



def clean_string_list(x_list):
    """Clean a list of words. Convet to lowercase and remove trailing and preceeeding spaces in words
    Parameters
    ----------
    x_list: list of strings. 
    	the strings are cleaned.
	Returns
	-------
	list of strings()

    """
    x_list=map((str.lower),x_list)
    x_clean=[i.lstrip() for i in x_list]
    x_clean=[i.rstrip() for i in x_clean]
    x_clean=filter(None, x_clean)
    return x_clean

def get_words(x, delimitter):
    """If a value is not Nan, returns tokenized words
    Parameters
    ----------
    x: str/Nan. 
    delimitter: character used to split the string
	Returns
	-------
	list of tokenized words

    """
    if str(x)!='nan':
        x=x.split(delimitter)        
        return x
    else:
        return ""

def get_sorted_list(x,order): #order=0-decreasing(similarity), 1-increasing(distance)
    x_df=pd.DataFrame(x)
    return list(x_df.sort_values(by=list(x_df.columns), ascending=order).index)

def getRecommendation(predicted_df, sort_order, window_len):
    predicted_df=predicted_df.T.apply(func=lambda x:get_sorted_list(x,sort_order),axis=0).T
    predicted_df.columns=range(predicted_df.shape[1])
    
    recommendation=pd.DataFrame({"reco":[""]*len(predicted_df)}, index=predicted_df.index)
    for ind in range(len(predicted_df.index)):
        recommendation["reco"][ind]=list(predicted_df.iloc[ind,0:window_len])

    return recommendation
