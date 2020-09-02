# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 18:14:46 2020

@author: mattias

##############################################################################
Predicts if a given peptide has antifungal activity (AFP) or not antifungal, but antimicrobial (NoAFPAMP).
5 fold cross validated metrics:
    Accuracy: 84 %
    Precision:
        AFP: 83 %
        NoAFPAMP: 29 %
    Recall: 
        AFP: 72 %
        NoAFPAMP: 67 % 
Model build by combining three individual models using an SVM:
    - A Neural Network(NN) trained on Mono-, Bi- and Trigram feature extraction from peptide sequence
    - A NN trained on dummy matrix of peptide sequence
    - A Random Forest Classifier trained on PseAAC data of peptide sequence
##############################################################################
"""
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import class_weight
from keras.models import Model, Sequential, model_from_json
import itertools
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from PseudoAAC import GetPseudoAAC

aminoAcids = ["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

def gen_ngram_vocabulary(ngram_range):
    vocab = []
    for i in range(ngram_range[0], ngram_range[1] + 1):
        for permutation in itertools.product(aminoAcids,repeat=i):
            vocab.append(''.join(permutation))
    return vocab


def gen_ngrams_data(string, ngram_range = (1,3)):
    #Transforms list of peptides to 2 and 3 gram matrix using tfidf or count 
    vocabulary = gen_ngram_vocabulary(ngram_range)
    vectorizer = CountVectorizer(ngram_range = ngram_range, lowercase=False, analyzer='char', vocabulary=vocabulary)
    ngrams = vectorizer.transform([string]).toarray()[0]

    return ngrams.tolist()
 
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

def keras_predict(peptides, kind):
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
    gen_data = {'NgramsModel':gen_ngrams_data, 'SeqModel':gen_sequential_data}[kind]
    json_file = open("../data/%s.json" %(kind), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("../data/%s_weights.h5"%(kind))
    scaler = pickle.load(open('../data/%s_scaler.pkl'%(kind), 'rb'))
    features = [gen_ngrams_data(peptide) for peptide in peptides]
    features = scaler.transform([gen_data(peptide) for peptide in peptides])
    
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
    PAACPF = pickle.load(open('../data/PseAACModel.sav', 'rb'))
    scaler = pickle.load(open('../data/PseAACModel_scaler.pkl', 'rb'))
    with open('../data/PseAACFeatures_ranking.csv','r') as file:
        ranking = np.array([int(x) for x in file.readline().split(',')])
        
    i = 200
    features = np.array([gen_PseAAC_data(peptide) for peptide in peptides])
    features = scaler.transform(features[:,ranking < i])
 
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
    svm = pickle.load(open('../data/CombinedModel.sav', 'rb'))
    features = np.concatenate((PAACRF_predict(peptides)[:,:-1], keras_predict(peptides, 'NgramsModel')[:,:-1], keras_predict(peptides, 'SeqModel')[:,:-1]), axis=1)
    predictions = svm.predict_proba(features)
    return predictions

if __name__ == '__main__':
    print(predict(['AAAAAAAAAAAAAAAAAAAAAAAAAA', 'AGAGAGAGAGAGAGAGAGAGAGAA']))