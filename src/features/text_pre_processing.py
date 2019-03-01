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



class TextPreProcessor:

    def __init__(self, path1, path2):
         self.path1 = path1
         self.path2 = path2

    def read_data(self):

        data = []
        with open(self.path1) as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                data.append(row)

        data = list(map(str, data))
        self.data = data
        return data

    def pre_processor(self):
        '''This function takes in a list of strings and performs a variety of text preprocessing functions.'''

        data = self.data
        # remove punctuatiom
        data = [re.sub(r'[^\w\s]', '', x) for x in data]
        # lower case
        data = [x.lower() for x in data]
        # remove numbers
        data = [re.sub(r'\d+', '', x) for x in data]
        # remove newlines '/n'
        data = [re.sub('\s+', ' ', x) for x in data]
        # remove non-ASCII characters
        data = [re.sub(r'[^\x00-\x7f]',r' ',x) for x in data]
        # remove underscores
        data = [re.sub(r'[_]', '', x) for x in data]
        # remove words less than 3 characters
        data = [re.sub(r'\b\w{1,2}\b', '', x) for x in data]
        self.data = data
        return data

    def sentence_to_words(self, data):
        '''This inputs a string of text and tokenizes words.'''

        data = self.data
        data_words = [word_tokenize(i) for i in data]
        self.data_words = data_words
        return data_words

    def remove_stopwords(self, data_words):
        data_words = self.data_words
        stop_words = stopwords.words('english')
        new_stop_words = ['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know',
                   'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see',
                   'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even',
                   'right', 'line', 'even', 'also', 'may', 'take', 'come', 'look', 'back', 'start', 'going',
                  'doing', 'what','whats', 'pron', 'dream', 'and']
        stop_words.extend(new_stop_words)

        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in data_words]

    def make_bigrams(self, data_words):

        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        return [bigram_mod[doc] for doc in data_words]

    def lemmatization(self, text):

        # Init the Wordnet Lemmatizer
        lemmatizer = WordNetLemmatizer()
        data_lemmatized = [lemmatizer.lemmatize(w) for w in text]
        return data_lemmatized


    def process_data(self):

        # Read in data
        raw_data = self.read_data()

        # preprocess data
        processed_data = self.pre_processor()

        # turn sentences into words
        data_words = self.sentence_to_words(processed_data)

        # Remove Stop Words
        data_words_nostops = self.remove_stopwords(data_words)

        # Form Bigrams
        data_words_bigrams = self.make_bigrams(data_words_nostops)
        data_words_bigrams = [item for sublist in data_words_bigrams for item in sublist]

        # Do lemmatization keeping only noun, adj, vb, adv

        data_lemmatized = self.lemmatization(data_words_bigrams)
        data_lemmatized = list(map(str, data_lemmatized))

        # remove words <= 1 character
        data_lemmatized = [re.sub(r'\b\w{1}\b', '', x) for x in data_lemmatized]
        self.data_lemmatized = data_lemmatized
        with open(self.path2, "w") as f:
            writer = csv.writer(f)
            writer.writerows(data_lemmatized)
        return data_lemmatized


if __name__ == '__main__':
    processor = TextPreProcessor(
       path1=sys.argv[1],
       path2=sys.argv[2],
    )

processor.process_data()


  # path for input file: '/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_4/data/external/sample_dream_2.csv'
# path for output file: '/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_4/data/external/processed_sample_dream_2.csv'