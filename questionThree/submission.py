# Enter your code here. Read input from STDIN. Print output to STDOUT
# Enter your code here. Read input from STDIN. Print output to STDOUT
import sys
import json
import numpy as np
from sklearn import ensemble

s = sys.stdin  # sets a variable that will access the input data
num_train_samples = int(s.readline())  # pull in first line (number of training rows)

train = []
for index in range(num_train_samples):  # iterate through the number of training lines
  train_line = json.loads(s.readline())
  train.append(train_line)
np.array(train)


num_test_samples = int(s.readline())  # get number of test samples
test = []
for index in range(num_test_samples):
  test.append(json.loads(s.readline()))
np.array(test)


ids = []
samples = []
classifiers = []
# simplified version (ignores textual data i.e. question text and topic name)
#
for row in train:
    ids.append(row[3])
    samples.append([int(row[1])]+[int(row[4])]+[int(row[5])]+[row[6]])
    classifiers.append(row[-1])

#convert booleans to 0/1
for row in samples:
    if row[3] == 'False':
        row[3] = 0
    else:
        row[3] = 1

#convert to numpy arrays
samples = np.array(samples, np.float)
classifiers = np.array(classifiers, np.float)

params = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'loss': 'huber', 'alpha': 0.95}
clf = ensemble.GradientBoostingRegressor(**params).fit(samples, classifiers)

prediction = clf.predict(test)


for index in range(num_test_samples):
  jsonString = json.JSONEncoder().encode({"__ans__": bool(prediction[index]), "question_key": data["question_key"]})
  sys.stdout.write(jsonString + "\n")
