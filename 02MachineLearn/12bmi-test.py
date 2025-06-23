from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd

tbl = pd.read_csv('resData/bmi.csv')

# 컬럼(열)을 자르고 정규화하기
label = tbl['label']
# 0~1 사이의 값으로 정규화
w = tbl['weight'] / 100
h = tbl['height'] / 200
# 정규화된 몸무게와 키를 데이ㅐ터프레임으로 변환
wh = pd.concat([w, h], axis=1)

# 학습 전용 데이터와 테스트 전용 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(wh, label)

clf = svm.SVC()
clf.fit(X_train, y_train)

predict = clf.predict(X_test)

ac_score = metrics.accuracy_score(y_test, predict)
cl_report = metrics.classification_report(y_test, predict)
print('정답률', ac_score)
print('리포트 =\n ', cl_report)