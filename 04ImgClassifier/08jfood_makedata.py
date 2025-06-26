from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import numpy as np

# 분류 대상 카테고리
root_dir = './download'
categories = ['Gyudon', 'Ramen', 'Sushi', 'Okonomiyaki', 'Karaage']
nb_classes = len(categories)
# 이미지 크기
image_size = 50

X = []
Y = []

for idx, cat in enumerate(categories):
    image_dir = root_dir + '/' + cat
    files = glob.glob(image_dir + '/*.jpg')

    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert('RGB')
        img = img.resize((image_size, image_size))
        # 이미지 픽셀값을 넘파이 배열로 변환
        data = np.asarray(img)
        # 넘파이 배열로 변환된 이미지와 카테고리 인덱스 추가
        X.append(data)
        Y.append(idx)

# 리스트를 ndarray로 변환
X = np.array(X)
Y = np.array(Y)

""" X_train : 훈령용(학습용) 입력 데이터
    X_test : 테스트용 입력 데이터
    Y_train : 훈련용 정답 레이블
    Y_test : 테스트용 정답 레이블 """
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

np.savez('./saveFiles/japanese_food.npz', X_train=X_train, X_test=X_test,
         Y_train=Y_train, Y_test=Y_test)
print('Task Finished..!!', len(Y))