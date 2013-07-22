import sys
import json
import numpy as np
from sklearn import metrics, cross_validation, linear_model

s = sys.stdin # sets a variable that will access the input data

num_train_lines = int(s.readline()) # pull in first line (number of training rows)
traindata = []
trainsubs = []
trainfeatures = []
for train_row_index in range(num_train_lines): # iterate through the number of training lines
    train_line = json.loads(s.readline()) # json parse the line
    topics = train_line['topics'] # get topics value
    if topics: # if there are topics
        tr = []
        tr.append(0) # tr 0: sum of all the followers of all topics
        #tr.append(len(topics)) # tr 1: number of topics
        while topics: # while there are topics remaining
            temp = topics.pop() # remove topic from topics
            #tr.append(temp['followers']) # add the followers for topic
            #tr.append(temp['name']) # add the name of topic
            tr[0] += temp['followers'] # add followers to sum of all topics followers
        #trainsubs.append(tr) # add to global var
        row = [] # make new row to temp store properties
        row.append(train_line["question_text"]) # row 0: question
        row.append(train_line["context_topic"]) # row 1: context_topic
        if train_line["context_topic"]: # check if this value is null
            row.append(train_line["context_topic"]["followers"]) # row 2: num ct followers 
            row.append(train_line["context_topic"]["name"]) # row 3: name of ct
        else: # append zeros to keep number of columns consistent
            row.append(0) # row 2: num ct followers
            row.append(0) # row 3: name of ct
        row.append(train_line["question_key"]) # row 4: question key
        row.append(train_line["anonymous"]) # row 5: anonymouse
        row.append(train_line["__ans__"]) # row 6: __ans__ which is dependent var
    #traindata.append(row) # add to global var
    trainfeatures.append([]) # add new empty array to global var
    trainfeatures[train_row_index].append(row[2]) # feature 0: number of context_topic followers
    trainfeatures[train_row_index].append(tr[0]) # feature 1: sum of topics followers
    trainfeatures[train_row_index].append(row[6]) # feature 2: dependent var

num_test_lines = int(s.readline()) # pull in next line (number of testing rows)
testdata = []
testsubs = []
testfeatures = []
for test_row_index in range(num_test_lines): # iterate through the number of testing lines
    test_line = json.loads(s.readline()) # json parse the line
    topics = test_line['topics'] # get topics value
    if topics: # if there are topics
        tr = []
        tr.append(0) # tr 0: sum of all the followers of all topics
        #tr.append(len(topics)) # tr 1: number of topics
        while topics: # while there are topics remaining
            temp = topics.pop() # remove topic from topics
            #tr.append(temp['followers']) # add the followers for topic
            #tr.append(temp['name']) # add the name of topic
            tr[0] += temp['followers'] # add followers to sum of all topics followers
        #testsubs.append(tr) # add to global var
        row = [] # make new row to temp store properties
        row.append(test_line["question_text"]) # row 0: question
        row.append(test_line["context_topic"]) # row 1: context_topic
        if test_line["context_topic"]: # check if this value is null
            row.append(test_line["context_topic"]["followers"]) # row 2: num ct followers 
            row.append(test_line["context_topic"]["name"]) # row 3: name of ct
        else: # append zeros to keep number of columns consistent
            row.append(0) # row 2: num ct followers
            row.append(0) # row 3: name of ct
        row.append(test_line["question_key"]) # row 4: question key
        row.append(test_line["anonymous"]) # row 5: anonymouse
    testdata.append(row) # add to global var
    testfeatures.append([]) # add new empty array to global var
    testfeatures[test_row_index].append(row[2]) # feature 0: number of context_topic followers
    testfeatures[test_row_index].append(tr[0]) # feature 1: sum of topics followers

### train the model here
# set constants
y = np.array(np.array(trainfeatures)[:,-1]) # set as last column (dependent var)
X = np.array(np.array(trainfeatures)[:,:-1]) # remove last column (dependent var)
num_features = X.shape[1]
num_train = X.shape[0]
model = linear_model.LogisticRegression()

# define functions
SEED = 503809822
def cv_loop(X, y, model, N):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                       X, y, test_size=.20,
                                       random_state = i*SEED)
        model.fit_transform(X_train, y_train)
        preds = model.predict_proba(X_cv)[:,1]
        auc = metrics.auc_score(y_cv, preds)
        #print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_auc += auc
    return mean_auc/N

# perform greedy feature selection
score_hist = []
N = 5 # number of cv_loop iterations
good_features = set([])
while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
    scores = []
    for f in range(num_features):
        if f not in good_features:
            score = cv_loop(X, y, model, N)
            scores.append((score, f))
            #print "Feature: %i Mean AUC: %f" % (f, score)
    good_features.add(sorted(scores)[-1][1])
    score_hist.append(sorted(scores)[-1])
    #print "Current features: %s" % sorted(list(good_features))
good_features.remove(score_hist[-1][1]) # Remove last added feature from good_features
good_features = sorted(list(good_features)) # Sorting

# Hyperparameter selection loop
score_hist = []
Cvals = np.logspace(-4, 4, 15, base=2)
for C in Cvals:
    model.C = C
    score = cv_loop(X, y, model, N)
    score_hist.append((score,C))
    #print "C: %f Mean AUC: %f" %(C, score)
bestC = sorted(score_hist)[-1][1]
#print "Best C value: %f" % (bestC)

# training the model
model.fit_transform(X, y)

### test model
#predictions = model.predict_proba(X)[:,1]
#print predictions

### run the test data here
Xtest = np.array(testfeatures)
predictions = model.predict_proba(Xtest)[:,1]


### output the responses here
for test_row_index in range(num_test_lines): # iterate through the number of testing lines
    if predictions[test_row_index] > 0.5:
        prediction = bool(1) # if greater than .5 set true
    else:
        prediction = bool(0) # if less than .5 set false

    jsonString = json.JSONEncoder().encode({
        "__ans__": prediction,
        "question_key": testdata[test_row_index][4]
    })
    sys.stdout.write(jsonString + "\n")
    