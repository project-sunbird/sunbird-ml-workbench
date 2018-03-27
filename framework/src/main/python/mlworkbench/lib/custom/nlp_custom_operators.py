import gensim
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
from gensim.models import LsiModel
import pandas as pd
import numpy as np
from gensim import corpora, models, similarities
import os
from mlworkbench.utils.common import create_directory, get_parent_dir


from mlworkbench.utils.nlp import to_unicode, get_sorted_list
from mlworkbench.lib.operation_definition import NodeOperation

def read_text_file_corpus(node):
    reader = NodeOperation(node)  # initiate
    reader.io_check((len(reader.inputs) == 1) & (len(reader.outputs) == 3))
    filepath = reader.graph_inputs[reader.inputs[0]]

    stop_words = set(stopwords.words('english'))
    stopchar = ["/n", "?", "!", ";", ":", ",", "-", "o'", '"', "."]
    filenames = os.listdir(filepath)
    if ".DS_Store" in filenames: filenames.remove(".DS_Store")
    filenames_full = [os.path.join(filepath, f) for f in filenames]
    text_list = []
    for file in (filenames_full):
        with open(file) as f:
            content = f.readlines()
            text = " ".join([line for line in content])
            for char in stopchar: text = text.replace(char, "")
            text = [word for word in to_unicode(text).lower().split() if word not in stop_words]

        text_list.append(text)

    dictionary = corpora.Dictionary(text_list)
    corpus = [dictionary.doc2bow(text) for text in text_list]
    create_directory(get_parent_dir(reader.graph_outputs[reader.outputs[0]]))
    with open(reader.graph_outputs[reader.outputs[2]], "w") as text_file:
        for item in filenames: text_file.write("%s\n" % item)

    # Save model, prediction, report
    dictionary.save(reader.graph_outputs[reader.outputs[0]])
    #reader.graph_outputs[reader.outputs[1]]
    corpora.MmCorpus.serialize(reader.graph_outputs[reader.outputs[1]], corpus)



def compute_doc_similarity(node):
    doc_sim = NodeOperation(node)  # initiate
    doc_sim.io_check((len(doc_sim.inputs) == 3) & (len(doc_sim.outputs) == 1))
    metadata_loc = doc_sim.graph_outputs[doc_sim.inputs[0]]  # get input data
    dict_loc = doc_sim.graph_outputs[doc_sim.inputs[1]]  # get input data
    corpus_loc = doc_sim.graph_outputs[doc_sim.inputs[2]]  # get input data

    with open(metadata_loc, 'r') as f:
        filenames = [line.rstrip() for line in f]

    dictionary = corpora.Dictionary.load(dict_loc)
    corpus = corpora.MmCorpus(corpus_loc)
    
    # Transform Text with TF-IDF
    tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model
    # corpus tf-idf
    corpus_tfidf = tfidf[corpus]
    ## pickle model here
    
    # compute simmilarity
    index = similarities.MatrixSimilarity(tfidf[corpus])
    sims = index[corpus_tfidf]
    similarity = list(sims)
    similarity_df = pd.DataFrame(similarity, index=filenames, columns=filenames)
    similarity_df.values[[np.arange(len(similarity_df))] * 2] = 0
    
    #doc_sim.put_dataframes([similarity_df], doc_sim.outputs[0])
    similarity_df.to_csv(doc_sim.graph_outputs[doc_sim.outputs[0]])  # write to file



def getRecommendation(predicted_df, sort_order, window_len):
    predicted_df=predicted_df.T.apply(func=lambda x:get_sorted_list(x,sort_order),axis=0).T
    predicted_df.columns=range(predicted_df.shape[1])
    
    recommendation=pd.DataFrame({"reco":[""]*len(predicted_df)}, index=predicted_df.index)
    for ind in range(len(predicted_df.index)):
        recommendation["reco"][ind]=list(predicted_df.iloc[ind,0:window_len])

    return recommendation

def doc_Recommendation(node): 
    eoc_reco = NodeOperation(node)  # initiate
    eoc_reco.io_check((len(eoc_reco.inputs) == 1) & (len(eoc_reco.outputs) == 1))
    sim_df=pd.read_csv(eoc_reco.graph_outputs[eoc_reco.inputs[0]],index_col=0)
    window=eoc_reco.arguments['num_reco']
    reco_df=getRecommendation(sim_df,0,window)
    reco_df.to_csv(eoc_reco.graph_outputs[eoc_reco.outputs[0]])

