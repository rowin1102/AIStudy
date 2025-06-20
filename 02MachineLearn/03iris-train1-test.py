from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import csv

def load_iris_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp)
        next(reader)
        for row in reader:
            try:
                features = list(map(float, row[0:4]))
                label = row[4]
                data.append((features, label))
            except ValueError:
                print("경고: 변환 실패로 건너뜬 줄:", row)
    return data

iris_data = load_iris_data('./resData/iris.csv')

X = [row[0] for row in iris_data]
y = [row[1] for row in iris_data]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=42, shuffle=True)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

ac_score = metrics.accuracy_score(y_test, pred)
print('정답률:', round(ac_score * 100, 2), '%')