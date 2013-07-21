# import os
import csv as csv
import numpy as np
import random
import math
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

columnNames = ['question_text', 'followers', 'name', 'question_key', 'promoted_to', 'num_answers', '__ans__']

dataFile = csv.reader(open('./questionThree/questionthree.csv', 'rb'))

#gets rid of header row
header = dataFile.next()

samples = []
classifiers = []
ids = []

#simplified version (ignores textual data i.e. question text and topic name)
for row in dataFile:
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


params = {'n_estimators': 110, 'max_depth': 3, 'learning_rate': 0.05, 'loss': 'huber', 'alpha': 0.95}
clf = GradientBoostingRegressor(**params).fit(samplesTrain, classifiersTrain)

mse = mean_squared_error(classifiersTest, clf.predict(samplesTest))
r2 = r2_score(classifiersTest, clf.predict(samplesTest))

print("MSE: %.4f" % mse)
print("R2: %.4f" % r2)

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, classifiersTrain in enumerate(clf.staged_decision_function(samplesTest)):
    test_score[i] = clf.loss_(classifiersTest, classifiersTrain)

plt.figure(figsize=(12, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-', label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-', label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')


feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(12, 6))
plt.subplot(1, 1, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
# plt.yticks(pos, samplesTrain.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
