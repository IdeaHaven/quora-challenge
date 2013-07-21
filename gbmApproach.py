import os
import csv as csv
import numpy as np
import random
 
columnNames = ['question_text', 'followers', 'name',
               'question_key', 'promoted_to', 'num_answers', '__ans__']

path = os.getcwd()

csv_file_object = csv.reader(open(path + '../test_parsed_from_bill.csv', 'rb'))

header = csv_file_object.next()

data = []
for row in csv_file_object:
    data.append(row)
data = np.array(data)

X = df[df.columns - [-1]]
Y = df[df.columns[-1]]
rows = random.sample(df.index, int(len(df)*.80))
x_train, y_train = X.ix[rows],Y.ix[rows]
x_test,y_test  = X.drop(rows),Y.drop(rows)

from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor
params = {'n_estimators': 500, 'max_depth': 6,
        'learn_rate': 0.1, 'loss': 'huber','alpha':0.95}
clf = GradientBoostingRegressor(**params).fit(x_train, y_train)

mse = mean_squared_error(y_test, clf.predict(x_test))
r2 = r2_score(y_test, clf.predict(x_test))
 
print("MSE: %.4f" % mse)
print("R2: %.4f" % r2)
