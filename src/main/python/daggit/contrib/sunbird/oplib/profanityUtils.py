import os
import json
import pandas as pd

def uncommonWords(A, B):
    count = {}
    for word in A.split():
        count[word] = count.get(word, 0) + 1
    for word in B.split():
        count[word] = count.get(word, 0) + 1

    return [word for word in count if count[word] == 1]

def betterProfanity(text):
    from better_profanity import profanity
    method="better_profanity"
    if profanity.contains_profanity(text):
        score=1.0
        censered=profanity.censor(text)
        loc=censered.find("****")

        detectedWord=UncommonWords(text, censered)
        try:
            detectedWord.remove("****")
        except:
            pass

        sentOfoccurance=" ".join(text[loc-50:loc+50].split(" ")[1:-1])
        timeOfoccurance='NA'
    else:
        score=0.0
        detectedWord= 'NA'
        sentOfoccurance= 'NA'
        timeOfoccurance='NA'
    return {"score":score,"method":method, "detectedWord":detectedWord,"sentOfoccurance":sentOfoccurance, "timeOfoccurance":timeOfoccurance}


def profanityFilter(text):
    from profanityfilter import ProfanityFilter

    method="ProfanityFilter"
    pf = ProfanityFilter()
    if pf.is_profane(text):
        score=1.0
        censered=pf.censor(text)
        loc=censered.find("****")

        detectedWord=UncommonWords(text, censered)
        try:
            detectedWord.remove("****")
        except:
            pass
        sentOfoccurance=" ".join(text[loc-50:loc+50].split(" ")[1:-1])
        timeOfoccurance='NA'
    else:
        score=0.0
        detectedWord= 'NA'
        sentOfoccurance= 'NA'
        timeOfoccurance='NA'
    return {"score":score,"method":method, "detectedWord":detectedWord,"sentOfoccurance":sentOfoccurance, "timeOfoccurance":timeOfoccurance}


def docProfanity(doc):
    from textblob import TextBlob
    from profanity_check import predict, predict_prob
    blob = TextBlob(doc)
    text=[]
    text_score=[]
    for sentence in blob.sentences:
        sub_text=str(sentence.string)
        text.append(sub_text)
        text_score.append(predict([sub_text])[0])
    return {'text':text, 'score':text_score}

def profanityCheck(text):
    method="profanity_check"
    score=0
    result = pd.DataFrame(docProfanity(text))
    if sum(list(result['score']))!=0:
        profane_sent=list(result[result['score']==1]['text'])
        score=1.0
        detectedWord = 'NA'
        sentOfoccurance =profane_sent
        timeOfoccurance='NA'

    else:
        profane_sent=""
        score=0.0
        detectedWord = 'NA'
        sentOfoccurance =profane_sent
        timeOfoccurance='NA'
    return {"score":score,"method":method, "detectedWord":detectedWord,"sentOfoccurance":sentOfoccurance, "timeOfoccurance":timeOfoccurance}

def text_profanity(text):
    check1 = betterProfanity(text)
    check2 = ProfanityFilter(text)
    check3 = profanityCheck(text)
    consolidated_score = (check1["score"]+check2["score"]+check3["score"])/3
    profanity_filter_op={"score":consolidated_score , "profanity":[check1, check2, check3]}
    return profanity_filter_op

