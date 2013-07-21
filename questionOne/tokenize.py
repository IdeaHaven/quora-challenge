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
    # items needed to test questions
# PANDAS USAGE
    question_tokens = nltk.word_tokenize(train.ix[row,0]) # all tokens (words, punc, etc)
    question_tags = nltk.pos_tag(question_tokens) # all tokens and parts of speech in lists
    question_first_word_tag = question_tags[0]    
    question_nouns = []
    question_verbs = []
    question_words = []
    for tag in question_tags: #separate parts of speech
        if tag[1][0] == "N":
            question_nouns.append(tag[0]) # all noun types 
        if tag[1][0] == "V":
            question_nouns.append(tag[0]) # all verb types 
        if tag[1] != ".":
            question_words.append(tag[0]) # all non punctuations types 
    # items needed to test questions
# PANDAS USAGE
    context_topic_tokens = nltk.word_tokenize(train.ix[row,2]) # all tokens (words, punc, etc)
    context_topic_tags = nltk.pos_tag(question_tokens) # all tokens and parts of speech in lists
    context_topic_nouns = []
    context_topic_words = []
    for tag in context_topic_tags: #separate parts of speech
        if tag[1][0] == "N":
            context_topic_nouns.append(tag[0]) # all noun types 
        if tag[1] != ".":
            context_topic_words.append(tag[0]) # all non punctuations types 

    # find number of nouns common between context_topic and question
    num_ctnoun_match_qnoun = 0    
    for ctnoun in context_topic_nouns:
        for qnoun in question_nouns:
            if ctnoun == qnoun:
                num_ctnoun_match_qnoun += 1
                
    # find number of words common between context_topic and question
    num_ctwords_match_qwords = 0    
    for ctword in context_topic_words:
        for qword in question_words:
            if ctword == qword:
                num_ctwords_match_qwords += 1
                
    # does first word match (Is..will..can..do..does..are..)            
                


# create features 
    features.append([]) # append a list to features for each row in data
# PANDAS USAGE
    features[row].append(train.ix[row,1])# # of followers in context topic
    features[row].append(len(question_tokens))# # of words in the question
    features[row].append()# # of topics
    features[row].append()# sum of followers in topics
# PANDAS USAGE
    features[row].append(1 if train.ix[1][5] else 0)# 1 for anon 0 for non-anon
    features[row].append(num_ctnoun_match_qnoun)# # of common nouns between question text and topics
    features[row].append(num_ctwords_match_qwords)# # of common words between question text and topics
    features[row].append()# does first word match (Is..will..can..do..does..are..)
    features[row].append()# What kind of question is it? (Who? What? Where? When? Why? How?)
    features[row].append()# no additional topics
    features[row].append()# question text count > 50
    features[row].append()# # of sentences
    features[row].append()# ends with a question mark
    features[row].append()# ratio of verbs
    features[row].append()# ratio of adjectives
    features[row].append()# What is the average length of word?
    features[row].append()# if words are capitalized after a period
    features[row].append()# Does the question have a proper noun in it?
    features[row].append()# Does the question have a name in it?
    features[row].append()# Does the question have a name of someone famous in it? (list of celebrities)
    features[row].append()# Is the question related to technology?
    #does the question contain the word "best"
    #Does the question contain the word "worst"
    #does the question contain the phrase "the most"
    #does the question contain the phrase "the least"
    #does the question contain the word "you"
    

