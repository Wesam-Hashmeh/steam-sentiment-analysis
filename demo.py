#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer #NLTK's porterstemmer 2.0 
import numpy

probPos = {}
with open('probPos.csv', mode='r') as f:
    data = csv.reader(f)
    for rows in data:
        if rows:
            probPos[rows[0]] = rows[1]
probNeg = {}
with open('probNeg.csv', mode='r') as f:
    data = csv.reader(f)
    for rows in data:
        if rows:
            probNeg[rows[0]] = rows[1]
    
 


# In[2]:


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


# In[3]:


def posFunc(sent):
    sent = sent.split()
    posSent = 0
    resultPos = []
    for word in sent:
        if word not in probPos:
            resultPos.append(((1) / (len(probPos) + len(probPos))))
        else:
            resultPos.append(probPos[word])
    resp = [float(ele) for ele in resultPos]
    posSent = numpy.prod(resp) 
    return posSent


def negFunc(sent):
    sent = sent.split()
    negSent = 0
    resultNeg = []
    for word in sent:
        if word not in probNeg:
            resultNeg.append(((1) / (len(probNeg) + len(probPos))))
        else:
            resultNeg.append(probNeg[word])
    resn = [float(ele) for ele in resultNeg]
    negSent = numpy.prod(resn) 
    return negSent


# In[ ]:


val = "Y"
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
    posCalc = posFunc(val)
    negCalc = negFunc(val)
    
    if posCalc > negCalc:
        print("was classified as Positive Review")
    else: 
        print("was classified as Negative Review")
    
    print(f" P( Positive_Review | {val_un} ) = {posCalc}")
    print(f" P( Negative_Review | {val_un} ) {negCalc} \n")
    
    val = input("Do you want to enter another sentence [Y/N]?")


# In[ ]:





# In[ ]:





# In[ ]:




