import os
import csv as csv
import numpy as np
import random
import math
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor

columnNames = ['question_text', 'followers', 'name',
               'question_key', 'promoted_to', 'num_answers', '__ans__']


dataFile = csv.reader(open('./questionThree/questionthree.csv', 'rb'))

# csv_file_object = csv.reader(open(path + '../test_parsed_from_bill.csv', 'rb'))

header = dataFile.next()

samples = []
classifiers = []
for row in dataFile:
    samples.append(row[:-1])
    classifiers.append(row[-1])

random.shuffle(samples)
random.shuffle(classifiers)

# define seperator between training (80%) and test data (20%)
sep = int(len(samples)*0.8)

samplesTrain = samples[:sep]
classifiersTrain = classifiers[:sep]

samplesTest = samples[sep:]
classifiersTrain = classifiers[sep:]



# data = np.array(data)



params = {'n_estimators': 500, 'max_depth': 6, 'learn_rate': 0.1, 'loss': 'huber', 'alpha': 0.95}
clf = GradientBoostingRegressor(**params).fit(samplesTrain, classifiersTrain)

mse = mean_squared_error(samplesTest, clf.predict(classifiersTest))
r2 = r2_score(samplesTest, clf.predict(classifiersTest))

print("MSE: %.4f" % mse)
print("R2: %.4f" % r2)
