import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

mr = pd.read_csv('./resData/mushroom.csv', header=None)
# 데이터 내부의 기호를 숫자로 변환
label = []
data = []
attr_list = []
for row_index, row in mr.iterrows():
    # 첫 번째 데이터는 라벨로 사용
    label.append(row.loc[0])
    row_data = []
    for v in row.loc[1:]:
        # 각 문자를 유니코드 값(정수)로 변환
        row_data.append(ord(v))
    data.append(row_data)

X_train, X_test, y_train, y_test = train_test_split(data, label)

# 데이터 학습시키기(랜덤 포레스트 사용)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

predict = clf.predict(X_test)

ac_score = metrics.accuracy_score(y_test, predict)
cl_report = metrics.classification_report(y_test, predict)
print('정답률', ac_score)
print('리포트 =\n ', cl_report)