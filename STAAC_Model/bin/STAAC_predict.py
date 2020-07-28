# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 18:14:46 2020

@author: matti

##############################################################################
Predicts if a given peptide has antifungal activity.
5 fold cross validated accuracy: 0.9
NOTE: The model can only identify AMPs without AF properties with a precision of \
    .25. A better model is in the process of being tuned. 
The model combines the results of a NN trained on the peptides sequential data \
    and mono, bi and trigrams with a RandomForestC trained on the peptides PseAAC
    (Type I, lamda = 2). The models' individual accuracies are: NN: 0.87, RFC: 0.88
##############################################################################
"""
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Model, Sequential, model_from_json
import itertools
import pandas as pd
import numpy as np
import pickle

from PseudoAAC import GetPseudoAAC

aminoAcids = ["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

def gen_ngram_vocabulary(ngram_range):
    vocab = []
    for i in range(ngram_range[0], ngram_range[1] + 1):
        for permutation in itertools.product(aminoAcids,repeat=i):
            vocab.append(''.join(permutation))
    return vocab


def peptide_to_ngram(string, ngram_range = (1,3)):
    #Transforms list of peptides to 2 and 3 gram matrix using tfidf or count 
    vocabulary = gen_ngram_vocabulary(ngram_range)
    vectorizer = CountVectorizer(ngram_range = ngram_range, lowercase=False, analyzer='char', vocabulary=vocabulary)
    ngrams = vectorizer.transform([string]).toarray()[0]
    
    ngrams_dict = {}
    for index,term in enumerate(vocabulary):
        if not term[::-1] in ngrams_dict:
            ngrams_dict[term] = ngrams[index]
        else:
            ngrams_dict[term[::-1]] += ngrams[index]
    
    return ngrams_dict
 
def gen_sequential_data(peptide):
    output = []
    for i in range(50):
        lst = [0 for _ in range(20)]
        if len(peptide) > i:
            lst[aminoAcids.index(peptide[i])] = 1
        output = output + lst
    return output



def gen_PseAAC_data(peptide):
    features = []
    AAPs = ['Hydrophobicity','hydrophilicity','residuemass','pK1','pK2','pI']   
    for j in range(1,7):
        for AAP in itertools.combinations(AAPs, j):
            pseaac = GetPseudoAAC(peptide, lamda=2,AAP=AAP)
            for val in pseaac:
                features.append(pseaac[val])
    return features

def gen_ngrams_data(peptide):
    features_dict = peptide_to_ngram(peptide)
    features = []
    for term in features_dict:
        features.append(features_dict[term])
    return features

def triseq_predict(peptides):
    """

    Parameters
    ----------
    peptides : list
        List of peptides to be assessed. Amino acids capitalized.

    Returns
    -------
    prediction : list
        List of shape (len(peptides), ) predicted probabillity that each given \
            peptide has antifungal properties.

    """
    json_file = open("../data/TriSeqModel.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    triseq = model_from_json(loaded_model_json)
    triseq.load_weights("../data/TriSeqModel_weights.h5")
    
    features_ngram = [gen_ngrams_data(peptide) for peptide in peptides]
    features_seq = [gen_sequential_data(peptide) for peptide in peptides]
    
    prediction = triseq.predict([features_ngram, features_seq])
    return prediction

def PAACRF_predict(peptides):
    """

    Parameters
    ----------
    peptides : list
        List of peptides to be assessed. Amino acids capitalized.

    Returns
    -------
    prediction : np.array(shape=(len(peptides), 2), dtype=float64)
        Predicted probabillity that each given peptide has antifungal properties.
        Format: [[p1(0), p1(1)],[p2(0), p2(1)],[p3(0),p3(1)], ...]

    """
    PAACPF = pickle.load(open('../data/PseAACModel.sav', 'rb'))
    with open('../data/PseAAC_feature_ranking_0723.csv','r') as file:
        ranking = np.array([int(x) for x in file.readline().split(',')])
        
    i = 500
    features = np.array([gen_PseAAC_data(peptide) for peptide in peptides])
    features = features[:,ranking < i]
    
    return PAACPF.predict_proba(features)

def predict(peptides):
    """
    Parameters
    ----------
    peptides : list
        List of peptides to be assessed. Amino acids capitalized.

    Returns
    -------
    prediction : np.array(shape=(len(peptides), 2), dtype=float64)
        Predicted probabillity that each given peptide has antifungal properties.
        Format: [[p1(0), p1(1)],[p2(0), p2(1)],[p3(0),p3(1)], ...]

    """
    sgd = pickle.load(open('../data/Combined_Model.sav', 'rb'))
    
    features = np.concatenate((PAACRF_predict(peptides)[:,[1]], triseq_predict(peptides)), axis=1)
    predictions = sgd.predict_proba(features)
    return predictions

