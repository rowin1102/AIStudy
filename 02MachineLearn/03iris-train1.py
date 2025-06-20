from sklearn import svm, metrics
import random, re


csv = []
with open('./resData/iris.csv', 'r', encoding='utf-8') as fp:
    for line in fp:
        # 줄바꿈 제거
        line = line.strip()
        # 컴마를 기준으로 문자열 분리
        cols = line.split(',')
        """ 람다식 : 문자열 데이터를 실수로 변환하는 기능으로 정의
                csv파일을 통해 읽어온 데이터는 문자이므로 실수로 변환하는게 필요함
                ^ : 문자열의 시작
                [0-9\.] : 숫자(0~9) 또는 마침표(.)가 한번 이상 나타나는 패턴 표현
                $ : 문자열의 끝
            즉, 123 혹은 12.34와 같은 문자와 매칭된다. """
        fn = lambda n : float(n) if re.match(r'^[0-9\.]+$', n) else n
        # cols 배열의 크기만큼 반복해서 fn을 호출한다. 이 결과를 리스트로 만들어준다.
        cols = list(map(fn, cols))
        # 리스트에 변환된 데이터를 추가한다.
        csv.append(cols)
# 첫 번째 줄의 헤더 제거(컬럼명이 입력되어 있음)
del csv[0]
random.shuffle(csv)
total_len = len(csv)

# 학습데이터(100개)와 테스트데이터(50개) 분할하기
""" 훈련에 사용하지 않은 데이터를 테스트에 활용해야 학습이 제대로 되었는지 확인할 수 있다.
    따라서 데이터를 이와 같이 분리해야 한다. """
train_len = int(total_len * 2 / 3)
train_data = []; train_label = []; test_data = []; test_label = []

for i in range(total_len):
    # 데이터와 라벨로 분리
    data = csv[i][0:4]
    label = csv[i][4]
    if i < train_len:
        train_data.append(data)
        train_label.append(label)
    else:
        test_data.append(data)
        test_label.append(label)

clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

ac_score = metrics.accuracy_score(test_label, pre)
print('정답률:', ac_score)