import nltk
import random
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
#nltk.download('all')
import string
import pandas as pd
import numpy as np

documents = [list((movie_reviews.words(fileid), category))
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

s = stopwords.words('english')

all_words = [w.lower() for w in movie_reviews.words()]
#print('Len with punctuation: ', len(all_words))
all_words_no_punc = [c for c in all_words if c not in string.punctuation]
#print('Len without punctuation: ', len(all_words_no_punc))
no_stopwords = [word for word in all_words_no_punc if word.lower() not in s]
#print('Len without stopwords: ', len(no_stopwords))

all_words = nltk.FreqDist(no_stopwords)

#We save 3000 most popular words
word_features = [w[0] for w in all_words.most_common(3000)]

#Returns a dictionary of the words in the document
# with True if word occur in the most popular words and False if not
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


#print(find_features(movie_reviews.words('neg/cv000_29416.txt')))


feature_sets = [(find_features(rev), category) for (rev, category) in documents]
results = []
for i in range(10):
    train_set = feature_sets[:1900]
    test_set = feature_sets[1900:]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    results.append(nltk.classify.accuracy(classifier, test_set) * 100)

    random.shuffle(feature_sets)

# classifier = nltk.NaiveBayesClassifier.train(train_set)
print("NBC accuracy: ", np.mean(results))
#
# classifier.show_most_informative_features(15)
