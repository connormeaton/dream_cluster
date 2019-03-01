import csv
import pickle
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.corpora.dictionary import Dictionary


def processed_csv_to_vec(sample_dream):
    '''This function inputs a filepath to a .csv file containing a preprocessed a sample dream. 
    It returns the probabilty distribution of that sample dream within each topic defined by the LDA model.'''
        
    split_sample = [d.split() for d in sample_dream]
    id2word = corpora.Dictionary(split_sample)
    sample_corpus = id2word.doc2bow(sample_dream)
    path = '/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_4/src/models/pickled_model_output/lda_model_file_11.sav'
    lda_model = pickle.load(open(path, 'rb'))

    vector = lda_model[sample_corpus]
    topics = []
    for j in vector:
        topics.append(vector)
    topics = topics[1][0][:11]    
    return topics


