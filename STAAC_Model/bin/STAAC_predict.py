# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 18:14:46 2020

@author: matti

##############################################################################
Predicts if a given peptide has antifungal activity.
5 fold cross validated metrics:
    Average precision: 0.84
    Average precision by label: AFP - 0.72, noAMP - 0.76, noAFPAMP - 0.65
Model build by combining three individual models using an SVM:
    Neural Network(NN) trained on Mono-, Bi- and Trigram feature extraction from peptide sequence:
        Average precision: 0.81
        Average precision by label: AFP - 0.62, noAMP - 0.88, noAFPAMP - 0.64
    NN trained on dummy matrix of peptide sequence:
        Average precision: 0.82
        Average precision by label: AFP - 0.66, noAMP - 0.86, noAFPAMP - 0.59
    Random Forest Classifier trained on PseAAC data of peptide sequence:
        Average precision: 0.86
        Average precision by label: AFP - 0.58, noAMP - 0.72, noAFPAMP - 0.83

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

def NN_predict(peptides, modelName):
    """
    Classifies peptides using a neural network
    
    Parameters
    ----------
    peptides : list
        List of peptides to be assessed. Amino acids capitalized.
    modelName : string
        Model to use to predict. Choose from 'NgramsModel', and 'SeqModel'
        

    Returns
    -------
    prediction : list
        List of shape (len(peptides), ) predicted probabillity that each given \
            peptide has antifungal properties.

    """
    data_generator = {'NgramsModel':gen_ngrams_data, 'SeqModel':gen_sequential_data}[modelName]
    json_file = open("../data/%s.json" %(modelName), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("../data/%s_weights.h5" %(modelName))
    
    features = np.array([data_generator(peptide) for peptide in peptides])
    
    prediction = model.predict(features)
    return prediction

def PAACRF_predict(peptides):
    """
    Classifies peptides using PseAAC and a RandomForestClassifier.
    
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
    PAACPF = pickle.load(open('../data/PseAACModel_test.sav', 'rb'))
    with open('../data/PseAAC_feature_ranking_0723.csv','r') as file:
        ranking = np.array([int(x) for x in file.readline().split(',')])
        
    i = 500
    features = np.array([gen_PseAAC_data(peptide) for peptide in peptides])
    features = features[:,ranking < i]
    
    return PAACPF.predict_proba(features)

def predict(peptides):
    """
    Combines models using an SVM classifier with a rbf kernel. \
        The SVM takes the predictions from the three classifiers in the order
        PseAAC, Trigrams, Sequential and only uses the first 2 probabillities 
        from each.
        
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
    svm = pickle.load(open('../data/Combined_Model_test.sav', 'rb'))
    pseaac_pred = PAACRF_predict(peptides)[:,:2]
    Trigram_pred = NN_predict(peptides, 'NgramsModel')[:,:2]
    Seq_pred = NN_predict(peptides, 'SeqModel')[:,:2]
    features = np.concatenate((pseaac_pred, Trigram_pred, Seq_pred), axis=1)
    predictions = svm.predict_proba(features)
    return predictions
