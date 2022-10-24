from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import os

import nltk # NLP
import re

import os

path1 = os.path.split(os.getcwd())[0] + '/project_datasets'

np.random.seed(0)

Fake_df = pd.read_csv(path1+'/'+'Fake.csv')
True_df = pd.read_csv(path1+'/'+'True.csv')


Fake_df['target'] = 0
True_df['target'] = 1

df = pd.concat([True_df, Fake_df], ignore_index=True, sort=False)
import random
df = df.sample(len(df),ignore_index=True)

df['text'] = df['title'] + ' ' + df['text']

df.drop(['title', 'subject', 'date',], axis=1, inplace=True)

# tokenize and cleaning

#  1. apply to remove url
def remove_url(text):
    return re.sub(r'http\S+', '', text)

df['text'] = df['text'].apply(remove_url)

#============================================

# 2. apply lower
def to_lower(text):
    return text.lower()

df['text'] = df['text'].apply(to_lower)

#============================================

# 3. apply contractions

# import sys
# import subprocess

# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'contractions'])

# subprocess.call("pip install --upgrade " + 'pip', shell=True)

import contractions

def remove_contractions(text):
    return ' '.join([contractions.fix(word) for word in text.split()])

df['text'] = df['text'].apply(remove_contractions)


#============================================

# 4. apply remove punctuations

def remove_punctuations(text):
    return re.sub('\[[^]]*\]', '', text)

df['text'] = df['text'].apply(remove_punctuations)

#============================================

# 5. apply to remove special characters

def remove_characters(text):
    return re.sub('[^a-zA-Z]', ' ', text)

df['text'] = df['text'].apply(remove_characters)

#============================================

# 6. apply to remove stopwords

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def remove_stopwords(text):
    return ' '.join([word for word in nltk.word_tokenize(text) if word not in stop_words])

df['text'] = df['text'].apply(remove_stopwords)

#============================================

# 7. apply stemming

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def stemming_words(text):
    return ' '.join(stemmer.stem(word) for word in text.split())

df['text'] = df['text'].apply(stemming_words)

#============================================

# 8. apply lemmatization

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split())

df['text'] = df['text'].apply(lemmatize_words)

df.to_csv(path1+'/'+'new_df.csv')
#============================================




