import numpy as np
import nltk
import csv

csvFile = csv.reader(open('./questionone.csv', 'rb'))
train = []

for row in csvFile:
    train.append(row)

train = np.array(train)


features = [] # make new empty list
counter = 0
for row in train:
# for row in range(len(train)-1): # for every row in the data
    # print row
    # data we need to create features
    # items needed to test questions
# PANDAS USAGE
    question_tokens = nltk.word_tokenize(row[0]) # all tokens (words, punc, etc)
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
            question_nouns[tag[0]] = tag[0] # all noun types
        elif tag[1][0] == "V":
            question_verbs[tag[0]] = tag[0] # all verb types
        elif tag[1] != ".":
            question_words[tag[0]] = tag[0] # all non punctuations types
        elif tag[1][0] == "J":
            question_adjectives[tag[0]] = tag[0] # all adjectives
    # itquestion_first_word_tagems needed to test questions
    context_topic_tokens = nltk.word_tokenize(train[:, 2]) # all tokens (words, punc, etc)
    context_topic_tags = nltk.pos_tag(question_tokens) # all tokens and parts of speech in lists
    context_topic_nouns = {}
    context_topic_words = {}
    for tag in context_topic_tags:  #separate parts of speech
        if tag[1][0] == "N":
            context_topic_nouns.has_key(tag[0])# all noun types
        elif tag[1] != ".":
            context_topic_words.has_key(tag[0]) # all non punctuations types

    # find number of nouns common between context_topic and question
    num_ctnoun_match_qnoun = 0
    for ctnoun in context_topic_nouns:
            if question_nouns[ctnoun]:
                num_ctnoun_match_qnoun += 1

    # find number of words common between context_topic and question
    num_ctwords_match_qwords = 0
    for ctword in context_topic_words:
            if question_words[ctword]:
                num_ctwords_match_qwords += 1

    # does first word match (Is..will..can..do..does..are..)

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
    features[counter].append(train[:, 1])# # of followers in context topic
    features[counter].append(len(question_tokens))# # of words in the question
#    features[counter].append()# # of topics
#    features[counter].append()# sum of followers in topics
# PANDAS USAGE
    features[counter].append(1 if row[5] else 0)# 1 for anon 0 for non-anon
    features[counter].append(num_ctnoun_match_qnoun)# # of common nouns between question text and topics
    features[counter].append(num_ctwords_match_qwords)# # of common words between question text and topics
#    features[counter].append()# does first word match (Is..will..can..do..does..are..)
#    features[counter].append()# What kind of question is it? (Who? What? Where? When? Why? How?)
#    features[counter].append()# no additional topics
#    features[counter].append()# question text count > 50
#    features[counter].append()# # of sentences
#    features[counter].append()# ends with a question mark
    features[counter].append(len(question_verbs)/len(question_tags))# ratio of verbs
    features[counter].append(len(question_adjectives)/len(question_tags))# ratio of adjectives
#    features[counter].append(question_correct_capitalization)# if words are capitalized after a period
    features[counter].append(1 if hasPNoun else 0)# Does the question have a proper noun in it?
#    features[counter].append()# Does the question have a name in it?
#    features[counter].append()# Does the question have a name of someone famous in it? (list of celebrities)
#    features[counter].append()# Is the question related to technology?
    counter += 1
print features