# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 16:34:41 2013

@author: gregory
"""

# PANDAS USAGE
import pandas as pd
#import numpy as np
import nltk

# PANDAS USAGE
train = pd.read_csv('questionone.csv')

features = [] # make new empty list
for row in range(train.shape[0]): # for every row in the data
    # data we need to create features
    # items needed to test questions
# PANDAS USAGE
    question_tokens = nltk.word_tokenize(train.ix[row,0]) # all tokens (words, punc, etc)
    question_tags = nltk.pos_tag(question_tokens) # all tokens and parts of speech in lists  
    question_nouns = {}
    question_verbs = {}
    question_words = {}
    question_adjectives = {}
    hasPNoun = False
    for tag in question_tags: #separate parts of speech
        if tag[1] == "NNP" or tag[1] == "NNPS":
            hasPNoun = True
        if tag[1][0] == "N":
            question_nouns[tag[0]] == tag[0] # all noun types
        elif tag[1][0] == "V":
            question_verbs[tag[0]] == tag[0] # all verb types
        elif tag[1] != ".":
            question_words[tag[0]] == tag[0] # all non punctuations types
        elif tag[1][0] == "J":
            question_adjectives[tag[0]] == tag[0] # all adjectives
    # itquestion_first_word_tagems needed to test questions
# PANDAS USAGE
    num_ctwords_match_qwords = 0
    num_ctnoun_match_qnoun = 0
    context_topic_tokens = nltk.word_tokenize(train.ix[row,2]) # all tokens (words, punc, etc)
    context_topic_tags = nltk.pos_tag(question_tokens) # all tokens and parts of speech in lists
    context_topic_nouns = {}
    context_topic_words = {}
    for tag in context_topic_tags:  #separate parts of speech
        if tag[1][0] == "N":
            context_topic_nouns[tag[0]] == tag[0]# all noun types
            if question_nouns[tag[0]]:
                num_ctnoun_match_qnoun += 1
        elif tag[1] != ".":
            context_topic_words[tag[0]] == tag[0] # all non punctuations types
            if question_words[tag[0]]:
                num_ctwords_match_qwords += 1

# check if question uses proper capitolization
#    question_correct_capitalization = 0
#    for i in range(len(question_tokens)):
#        if question_tokens[i] == ".":
#            if question_tokens[i + 1][0] and question_tokens[i + 1][0].isupper():
#                question_correct_capitalization += 1

# check if question contains a proper noun

# create features
    features.append([]) # append a list to features for each row in data
# PANDAS USAGE
    features[row].append(train.ix[row,1])# # of followers in context topic
    features[row].append(len(question_tokens))# # of words in the question
#    features[row].append()# # of topics
#    features[row].append()# sum of followers in topics
# PANDAS USAGE
    features[row].append(1 if train.ix[1][5] else 0)# 1 for anon 0 for non-anon
    features[row].append(num_ctnoun_match_qnoun)# # of common nouns between question text and topics
    features[row].append(num_ctwords_match_qwords)# # of common words between question text and topics
#    features[row].append()# does first word match (Is..will..can..do..does..are..)
#    features[row].append()# What kind of question is it? (Who? What? Where? When? Why? How?)
#    features[row].append()# no additional topics
#    features[row].append()# question text count > 50
#    features[row].append()# # of sentences
#    features[row].append()# ends with a question mark
    features[row].append(len(question_verbs)/len(question_tags))# ratio of verbs
    features[row].append(len(question_adjectives)/len(question_tags))# ratio of adjectives
#    features[row].append(question_correct_capitalization)# if words are capitalized after a period
    features[row].append(1 if hasPNoun else 0)# Does the question have a proper noun in it?
#    features[row].append()# Does the question have a name in it?
#    features[row].append()# Does the question have a name of someone famous in it? (list of celebrities)
#    features[row].append()# Is the question related to technology?
print features