from sklearn import svm, metrics
import pandas as pd

xor_input = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

xor_df = pd.DataFrame(xor_input)
# 학습데이터와 레이블데이터를 분리
X_train = xor_df.loc[:, 0:1]
y_train = xor_df.loc[:, 2]
print('X_train\n', X_train)

# 데이터 학습 및 예측
clf = svm.SVC()
clf.fit(X_train, y_train)
X_test = X_train
pred = clf.predict(X_test)

ac_score = metrics.accuracy_score(y_train, pred)
print('정답률 :', ac_score)
""" 사이킷런에서 제공하는 내장함수를 통해 프로그램을 간단히 작성할 수 있다. """