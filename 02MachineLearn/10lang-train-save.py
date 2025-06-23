from sklearn import svm
import joblib, json

# 각 언어의 출현 빈도 데이터(JSON) 읽어 들이기
with open('lang/freq.json', 'r', encoding='utf-8') as fp:
    d = json.load(fp)
    data = d[0]

clf = svm.SVC()
# 알파벳의 출현 빈도수와 레이블을 통해 fit 함수 실행으로 학습
clf.fit(data['freqs'], data['labels'])

# 학습 데이터 저장하기
joblib.dump(clf, 'lang/freq.pkl')
print('ok')