import nltk
import random
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import string
import pandas as pd
import numpy as np

documents = [list((movie_reviews.words(fileid), category))
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

s = stopwords.words('english')

all_words = [w.lower() for w in movie_reviews.words()]
print('Len with punctuation: ', len(all_words))
all_words_no_punc = [c for c in all_words if c not in string.punctuation]
print('Len without punctuation: ', len(all_words_no_punc))
no_stopwords = [word for word in all_words_no_punc if word.lower() not in s]
print('Len without stopwords: ', len(no_stopwords))

all_words = nltk.FreqDist(no_stopwords)
print(all_words.most_common(15))

word_features = list(all_words.keys())[:3000]