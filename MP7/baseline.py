"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

import numpy as np

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    track = {}
    tot_track = {}
    
    for sentence in train:
        for pairs in sentence:
            words = pairs[0]
            tags = pairs[1]
            if words in track.keys():
                tags_tracked = track[words]
                if tags in tags_tracked.keys():
                    tags_tracked[tags] += 1
                else:
                    tags_tracked[tags] = 1
            else:
                track[words] = {tags:1}
            
            if tags in tot_track.keys():
                tot_track[tags] += 1
            else:
                tot_track[tags] = 1
    
    default_tags = max(zip(tot_track.values(), tot_track.keys()))[1]
    retval = []
    for sentence in test:
        found = []
        for words in sentence:
            if words in track.keys():
                found.append((words, max(zip(track[words].values(), track[words]. keys()))[1]))
            else:
                found.append((words, default_tags))
        
        retval.append(found)
    
    return retval