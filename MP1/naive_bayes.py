# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter
import numpy as np


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naiveBayes(dev_set, train_set, train_labels, laplace=0.015, pos_prior=0.9, silently=False):
    print_values(laplace,pos_prior)

    yhats = []
    
    neg_prior = 1-pos_prior
    
    known_pos, unknown_pos = training(train_set, train_labels, laplace, 1)
    known_neg, unknown_neg = training(train_set, train_labels, laplace, 0)
    
    for ind_set in dev_set:
        prob_negwords = 0
        prob_poswords = 0
        
        prob_poswords += np.log(pos_prior)
        prob_negwords += np.log(neg_prior)
        
        for words in ind_set:
            if words in known_pos:
                prob_poswords += np.log(known_pos[words])
            else:
                prob_poswords += np.log(unknown_pos)
        
            if words in known_neg:
                prob_negwords += np.log(known_neg[words])
            else:
                prob_negwords += np.log(unknown_neg)
        
        if prob_poswords >+ prob_negwords:
            yhats.append(1)
        else:
            yhats.append(0)
    return yhats


def training (train_set, train_labels, laplace_smoothing, type):
    nlabels = len(train_labels)
    word_list = {}
    known_prob = {}
    unknown_prob = 0
    nwords = 0
    ntypes = 0
    
    for i in range(nlabels):
        if (train_labels[i] == type):
            for training_word in train_set[i]:
                if training_word in word_list:
                    word_list[training_word] += 1
                else:
                    word_list[training_word] = 1
    
    ntype = len(word_list)
    
    for i in word_list:
        nwords += word_list[i]
        
    for i in word_list:
        known_prob[i] = (word_list[i] + laplace_smoothing)/(nwords + (laplace_smoothing * (ntype+1)))
        
        unknown_prob = laplace_smoothing/(nwords + (laplace_smoothing*(ntypes + 1)))
    
    return known_prob, unknown_prob