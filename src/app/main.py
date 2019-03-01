import pickle
import pandas as pd
import csv


#tf-idf model and cosine similary querying
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
import gensim.corpora as corpora

# import local models
from SampleTextPreprocessor import pre_processor
from predict_LDA import processed_csv_to_vec

def get_dream():
    # collect dream and process text
    input_dream = pre_processor()
    # find probability distribution for new dream in LDA topics
    topic_prob_output = processed_csv_to_vec(input_dream)
    print( f'\nThe probabilties of your dream to be within the following topics: \n{topic_prob_output}')
    # prep input dream for tfidf vectorizer
    input_dream = ' '.join(input_dream)
    return input_dream

input_dream = get_dream()
          
          
def read_corpus(path):
   
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data_words_nostops = list(reader)
    
    stringified_corpus = [' '.join(i) for i in data_words_nostops]
    return stringified_corpus
    
stringified_corpus = read_corpus('/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_4/data/processed_data/processed_dreams.csv'
)

def raw_dream_corpus():
    path_to_raw_dreams = "/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_4/data/raw/final_dataframe.csv"
    df = pd.read_csv(path_to_raw_dreams)
    raw_dreams = df.selftext.values.tolist()
    return raw_dreams

raw_dreams = raw_dream_corpus()

def return_similar_docs(training_corpus, new_doc):

    # create tfidf vectorizer
    tfidf = TfidfVectorizer()
    # fit tfidf vectorizer on entire corpus
    queryTFIDF = tfidf.fit_transform(training_corpus)
    # get tfidf weights for the new_doc terms
    sample_query = tfidf.transform([new_doc])
    # create array of cosine similarities of the new_doc terms with each document from entire corpus
    cosine_similarities = linear_kernel(sample_query, queryTFIDF).flatten()
    # return index position of top 5 most similar documents in corpus to the new_doc
    related_docs_indices = cosine_similarities.argsort()[:-5:-1]
    print(f'\nThe 5 most similar documents are: {related_docs_indices}')
    # return cosine similiarty of top 5 most similar documents in corpus to the new_doc
    related_docs_similiarity = cosine_similarities[cosine_similarities.argsort()[:-5:-1]]
    print(f'\nThe cosine similarites of the 5 most similar documents are: {related_docs_similiarity}\n')
    
    print(f'The most similar dream to yours is:\n {raw_dreams[related_docs_indices.tolist()[0]]}\n\n')
    print(f'The second most similar dream to yours is:\n {raw_dreams[related_docs_indices.tolist()[1]]}\n\n')
    print(f'The third most similar dream to yours is:\n {raw_dreams[related_docs_indices.tolist()[2]]}\n\n')

    
return_similar_docs(stringified_corpus, input_dream)
