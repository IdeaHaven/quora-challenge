# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 16:34:41 2013

@author: gregory
"""

# PANDAS USAGE
import pandas as pd
import numpy as np
import nltk

# PANDAS USAGE
train = pd.read_csv('questionone.csv')

features = [] # make new empty list
for row in range(train.shape[0]): # for every row in the data
    # data we need to create features
    question_tokens = nltk.word_tokenize(train.ix[1,0]) # all tokens (words, punc, etc)
    question_tags = nltk.pos_tag(question_tokens) # all tokens and parts of speech
    question_first_word_tag = question_tags[0]    
    question_nouns = []
    question_verbs = []
    for tag in question_tags: #separate parts of speech
        if tag[1][0] == "N":
            question_nouns.append(tag[0]) # all noun types 
        if tag[1][0] == "V":
            question_nouns.append(tag[0]) # all verb types 

# create features 
    features.append([]) # append a list to features for each row in data
    features[row].append(train.ix[row,1])# # of followers in context topic
    features[row].append(len(question_tokens))# # of words in the question
    features[row].append()# # of topics
# PANDAS USAGE
    features[row].append(train.ix[1][5])# sum of followers in topics
    features[row].append()# not anon
    features[row].append()# # of common nouns between question text and context topic
    features[row].append()# # of common nouns between question text and topics
    features[row].append()# Is it a yes or no question? (Is..will..can..do..does..are..)
    features[row].append()# What kind of question is it? (Who? What? Where? When? Why? How?)
    features[row].append()# no additional topics
    features[row].append()# question text count > 50
    features[row].append()# # of sentences
    features[row].append()# ends with a question mark
    features[row].append()# freq of punctuation
    features[row].append()# ratio of extraneous pronouns
    features[row].append()# ratio of verbs
    features[row].append()# ratio of adjectives
    features[row].append()# What is the average length of word?
    features[row].append()# if words are capitalized after a period
    features[row].append()# Does the question have a proper noun in it?
    features[row].append()# Does the question have a name in it?
    features[row].append()# Does the question have a name of someone famous in it? (list of celebrities)
    features[row].append()# Is the question related to technology?

for tag in question_tags:
    if tag[1] == "NNP":
        print "Proper Noun"
