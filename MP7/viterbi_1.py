"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    init_prob['START'] = 1
    
    tprob_count = defaultdict(lambda:defaultdict(lambda:0))
    eprob_count = defaultdict(lambda:defaultdict(lambda:0))
    
    for sentence in sentences:
        for i, pair in enumerate(sentence):
            words = pair[0]
            tag = pair[1]

            if (i < (len(sentence) - 1)):
                ftag = sentence[1+i][1]
                tprob_count[tag][ftag]+=1
                eprob_count[tag][words]+=1
            else:
                eprob_count[tag][words]+=1
            
    for tag in eprob_count.keys():
        total = sum(eprob_count[tag].values())
        length = len(eprob_count[tag])
        denominator = ((total+1)*emit_epsilon) + length
        
        for words in eprob_count[tag].keys():
            emit_prob[tag][words] = eprob_count[tag][words] + emit_epsilon
            (emit_prob[tag][words])/denominator
            if (emit_prob[tag][words] == 0):
                emit_prob[tag][words] = emit_epsilon
            
        emit_prob[tag]['UNKNOWN'] = emit_epsilon / denominator

    for tag in tprob_count.keys():
        total = sum(eprob_count[tag].values())
        length = len(eprob_count[tag])
        denominator = ((total+1)*emit_epsilon) + length
        
        for translation in tprob_count[tag].keys():
            trans_prob[tag][translation] = tprob_count[tag][translation] + emit_epsilon
            (trans_prob[tag][translation])/denominator
            
    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.
    tagset= list(emit_prob.keys())

    if (i == 0):
        for tag in tagset:
            if word in emit_prob[tag]:
                eprob_set = [log(emit_prob[tag][word])]
            else:
                eprob_set = [log(emit_epsilon)]
            predict_tag_seq[tag] = [tag]
        log_prob = dict(zip(tagset, eprob_set))
        return log_prob, predict_tag_seq
  
    for tag in tagset:
        if word in emit_prob[tag]:
            etag = word
        else:
            etag = 'UNKNOWN'
        
        if etag == 'UNKNOWN':
            default_prob = log(emit_epsilon)
        else:
            default_prob = emit_prob[tag][word]
        
        max_track = float('-inf')
        tag_track = ""
        for i in prev_prob:
            if i in trans_prob and tag in trans_prob[i]:
                tprob = log(trans_prob[i][tag])
            else:
                tprob = log(emit_epsilon)
            
            if tag in emit_prob and etag in emit_prob[tag]:
                emits = log(emit_prob[tag][etag])
            else:
                emits = log(default_prob)
            
            prob_total = tprob + emits + prev_prob[i]
            
            if prob_total > max_track:
                max_track = prob_total
                tag_track = i
        
        log_prob[tag] = max_track
        
        if tag_track in prev_predict_tag_seq:
            predict_tag_seq[tag] = prev_predict_tag_seq[tag_track]
        else:
            predict_tag_seq[tag] = []
        predict_tag_seq[tag] = predict_tag_seq[tag] + [tag]
    
    return log_prob, predict_tag_seq


def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        val_max = float('-inf')
        key_max = None
        
        for key, value in log_prob.items():
            if value > val_max:
                val_max = value
                key_max = key
                
        predicted = predict_tag_seq[key_max]
        
        pair = []
        for word, tag in zip(sentence, predicted):
            pair.append((word, tag))
        predicts.append(pair)
    return predicts