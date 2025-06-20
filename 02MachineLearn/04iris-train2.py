from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd

csv = pd.read_csv('./resData/iris.csv')

X = csv.iloc[:, 0:4]
y = csv.iloc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = svm.SVC()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

ac_score = metrics.accuracy_score(y_test, pred)
print('정답률:', ac_score)