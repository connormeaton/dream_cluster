import sys
import csv
# preprocessing
import gensim
from gensim.utils import simple_preprocess
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# lemmitization
from nltk.stem import WordNetLemmatizer



def pre_processor():
    
    user_input = input('Please enter a dream: ')
    data = user_input

    # remove punctuatiom
    data = re.sub(r'[^\w\s]', '', data)
    # lower case
    data = data.lower()
    # remove numbers
    data = re.sub(r'\d+', '', data)
    # remove newlines '/n'
    data = re.sub('\s+', ' ', data)
    # remove non-ASCII characters
    data = re.sub(r'[^\x00-\x7f]',r' ',data)
    # remove underscores
    data = re.sub(r'[_]', '', data) 
    # remove words less than 3 characters
    data = re.sub(r'\b\w{1,2}\b', '', data)
       
    # create stop_words
    stop_words = stopwords.words('english')
    new_stop_words = ['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know',
               'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see',
               'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even',
               'right', 'line', 'even', 'also', 'may', 'take', 'come', 'look', 'back', 'start', 'going',
              'doing', 'what','whats', 'pron', 'dream', 'and']
    stop_words.extend(new_stop_words)

    # remove stop words and tokenize
    data_words = [i for i in word_tokenize(data.lower()) if i not in stop_words] 
    
    # create bigrams
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    data_words_bigrams = bigram_mod[data_words]

    # Init the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    data_lemmatized = [lemmatizer.lemmatize(w) for w in data_words_bigrams]
    return data_lemmatized

