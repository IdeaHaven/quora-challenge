# Enter your code here. Read input from STDIN. Print output to STDOUT
# Enter your code here. Read input from STDIN. Print output to STDOUT
import sys
import json
import numpy as np
import random
import math
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import ensemble

train = []

np.array(train)

count = 0
trainCount = 0
for line in sys.stdin:
    try:
        data = json.loads(line)
        if int(data):
            count = count + 1
    except:
        try:
            row = [data["question_text"], 
         data["context_topic"]["followers"], 
	       data["context_topic"]["name"], 
	       data["question_key"],
	       data["promoted_to"],
	       data["num_answers"],
	       data["anonymous"],
	       data["__ans__"]]
            np.array(row)
            train.append(row)
            trainCount = trainCount + 1
        except:
            pass
        samples = []
        classifiers = []
        ids = []
        
        #simplified version (ignores textual data i.e. question text and topic name)
        for row in train:
            ids.append(row[3])
            samples.append([int(row[1])]+[int(row[4])]+[int(row[5])]+[row[6]])
            classifiers.append(row[-1])
        
        # full version
        # for row in dataFile:
        #     samples.append(row[:-1])
        #     classifiers.append(row[-1])
        
        #convert booleans to 0/1
        for row in samples:
            if row[3] == 'False':
                row[3] = 0
            else:
                row[3] = 1
        
        # define seperator between training (80%) and test data (20%),
        # partition samples/classifiers
        sep = int(len(samples)*0.8)
        
        samplesTrain = samples[:sep]
        classifiersTrain = classifiers[:sep]
        
        samplesTest = samples[sep:]
        classifiersTest = classifiers[sep:]
        
        #convert to numpy arrays
        samplesTrain = np.array(samplesTrain, np.float)
        classifiersTrain = np.array(classifiersTrain, np.float)
        
        samplesTest = np.array(samplesTest, np.float)
        classifiersTest = np.array(classifiersTest, np.float)
        
        
        params = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'loss': 'huber', 'alpha': 0.95}
        clf = ensemble.GradientBoostingRegressor(**params)
    if count == 2:
        data = json.loads(line)
        try:
            if int(data):
                pass
            else:
                print data
        except:
            try:
                samplesTest = [data["question_text"], 
               data["context_topic"]["followers"], 
               data["context_topic"]["name"], 
               data["question_key"],
               data["promoted_to"],
               data["num_answers"],
               data["anonymous"],
               data["__ans__"]]
                prediction = clf.predict(samplesTest)
                jsonString = json.JSONEncoder().encode({
                    "__ans__": prediction, 
                    "question_key": data["question_key"]
                })
                sys.stdout.write(jsonString + "\n")
            except:
                pass
