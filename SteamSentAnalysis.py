#!/usr/bin/env python
# coding: utf-8

# In[2]:


# MAIN 
import sys #sys.argv[1] This is to check if preproccesing step to be ignored or not
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer #NLTK's porterstemmer 2.0 
import matplotlib.pyplot as plt
import random as rn

print("Training classifier…\n")

data = pd.read_csv('dataset.csv')   
df = pd.DataFrame(data, columns=['review_text', 'review_score'])
dfTrain = df[0:5120000].copy()
dfTest =  df[5120000:6400000].copy()



# In[3]:


# Pre Processessing
#used the following resources : https://www.kaggle.com/code/sudalairajkumar/getting-started-with-text-preprocessing
#                               https://www.kaggle.com/code/danielbeltsazar/steam-games-reviews-analysis-sentiment-analysis
#Used it mainly from this source ^ 

def clean(raw):
    result = re.sub("<[a][^>]*>(.+?)</[a]>", 'Link.', str(raw))
    result = re.sub('&gt;', "", result)
    result = re.sub('&#x27;', "'", result)
    result = re.sub('&quot;', '"', result)
    result = re.sub('&#x2F;', ' ', result)
    result = re.sub('<p>', ' ', result)
    result = re.sub('</i>', '', result)
    result = re.sub('&#62;', '', result)
    result = re.sub('<i>', ' ', result)
    result = re.sub("\n", '', result)
    return result

def remove_num(texts): 
   output = re.sub(r'\d+', '', texts)
   return output



def remove_Emoji(x):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'', x)

def remove_symbols(x):
    cleaned_string = re.sub(r"[^a-zA-Z0-9?!.,]+", ' ', x)
    return cleaned_string

def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"',','))
    return final

stop=set(stopwords.words("english"))
def remove_stopword(text):
   text=[word.lower() for word in text.split() if word.lower() not in stop]
   return " ".join(text)



def Stemming(text):
   stem=[]
   stopword = stopwords.words('english')
   snowball_stemmer = SnowballStemmer('english')
   word_tokens = nltk.word_tokenize(text)
   stemmed_word = [snowball_stemmer.stem(word) for word in word_tokens]
   stem=' '.join(stemmed_word)
   return stem


def preproc(df,review):
    df[review] = df[review].apply(clean)
    if sys.argv[1] != 'YES':
        df[review] = df[review].apply(remove_Emoji)
    df[review] = df[review].str.lower()
    df[review] = df[review].apply(remove_num)
    df[review] = df[review].apply(remove_symbols)
    df[review] = df[review].apply(remove_punctuation)
    df[review] = df[review].apply(remove_stopword)
    df[review] = df[review].apply(Stemming)
    return df


# In[4]:


dfTrain = preproc(dfTrain,'review_text')


# In[5]:



posTrain = []
for i in range(len(dfTrain["review_score"])):
    if (dfTrain['review_score'].iloc[i])  == 1:
        posTrain.append( (dfTrain['review_text'].iloc[i])) 

        
negTrain = []
for i in range(len(dfTrain["review_score"])):
    if (dfTrain['review_score'].iloc[i])  == -1:
        negTrain.append( (dfTrain['review_text'].iloc[i])) 
        
vocab = []
for i in range(len(dfTrain["review_score"])):
    vocab.append( (dfTrain['review_text'].iloc[i])) 


# In[6]:


temp = dfTrain.groupby('review_score').count()['review_text'].reset_index().sort_values(by='review_text',ascending=False)


# In[7]:



def tokenize(word):
    return word 

tokenPosTrain = [tokenize(word) for sentence in posTrain for word in sentence.split()]
tokenNegTrain = [tokenize(word) for sentence in negTrain for word in sentence.split()]
tokenVocab = [tokenize(word) for sentence in vocab for word in sentence.split()]


# In[8]:


posTrain_freq = nltk.FreqDist(tokenPosTrain)
negTrain_freq = nltk.FreqDist(tokenNegTrain)
vocab_freq = nltk.FreqDist(tokenVocab)

countPos = posTrain_freq.most_common(posTrain_freq.B()) 
countNeg = negTrain_freq.most_common(negTrain_freq.B()) 
countVocab = vocab_freq.most_common(vocab_freq.B()) 


# In[9]:


probPos = {}
probNeg = {}

for word in countPos:
    addOneProb =  ((word[1] + 1) / (len(tokenPosTrain) + len(countVocab)))
    probPos[word[0]] = addOneProb

for word in countNeg:
    addOneProb =  ((word[1] + 1) / (len(tokenNegTrain) + len(countVocab)))
    probNeg[word[0]] = addOneProb
    


# In[29]:


def rowCalc(sent):
    sent = sent.split()
    posSent, negSent = 0, 0
    resultPos, resultNeg = [], []
    for word in sent:
        if word not in probPos:
            resultPos.append(((1) / (len(tokenPosTrain) + len(countVocab))))
        if word not in probNeg:
            resultNeg.append(((1) / (len(tokenNegTrain) + len(countVocab))))
        if word in probPos:
            resultPos.append(probPos[word])
        if word in probNeg:
            resultNeg.append(probNeg[word])
    posSent = np.prod(resultPos) 
    negSent = np.prod(resultNeg) 
    return posSent, negSent


# In[30]:


#TESTING PART
print("Testing classifier…")

dfTest = preproc(dfTest,'review_text')

truePos, trueNeg, falsePos, falseNeg = 0, 0, 0, 0
classNegProb = len(negTrain)/len(vocab)
classPosProb = len(posTrain)/len(vocab)



testingClass = 1

for index, row in dfTest.iterrows():
    posCalc, negCalc = 0, 0
    posSent, negSent = rowCalc(row['review_text'])
    posCalc = classPosProb * posSent
    negCalc = classNegProb * negSent
    if posCalc > negCalc:
        testingClass = 1
    else: 
        testingClass = -1

    if testingClass == 1 and row['review_score'] == 1:
        truePos += 1
    if testingClass == 1 and row['review_score'] == -1:
        falsePos +=1
    if testingClass == -1 and row['review_score'] == 1:
        falseNeg += 1
    if testingClass == -1 and row['review_score'] == -1:
        trueNeg += 1
   


# In[31]:


print("Test results / metrics:")
print("Number of true positives: ", truePos)
print("Number of true negatives: ", trueNeg)
print("Number of false positives: ", falsePos)
print("Number of false negatives: ", falseNeg)
sensi = truePos / (truePos + falseNeg)
print("Sensitivity (recall): ", sensi)
speci = trueNeg / (trueNeg + falsePos)
print("Specificity: ", speci)
perci = truePos / (truePos + falsePos)
print("Precision: ", perci)
negpred = trueNeg / (trueNeg + falseNeg)
print("Negative predictive value: ", negpred)
acc = (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)
print("Accuracy: ", acc )
fscore = truePos / (truePos + .5*(falsePos + falseNeg))
print("F-score: ",fscore)


# In[28]:


val = "Y"
print("Enter your sentence: ")
while val.upper() == "Y":
    val = input("Enter your Sentence: ")
    val_un = val
    posCalc, negCalc = 0, 0
    val = val.lower()
    val = remove_num(val)
    val = remove_symbols(val)
    val = remove_punctuation(val)
    val = remove_stopword(val)
    val = Stemming(val)
    posCalc, negCalc = rowCalc(val)
    
    if posCalc > negCalc:
        print("was classified as Positive Review")
    else: 
        print("was classified as Negative Review")
    
    print(f" P( Positive_Review | {val_un} ) = {posCalc}")
    print(f" P( Negative_Review | {val_un} ) {negCalc} \n")
    
    val = input("Do you want to enter another sentence [Y/N]?")

