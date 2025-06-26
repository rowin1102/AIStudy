from tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import utils
import numpy as np

# root_dir = './download'
categories = ['Gyudon', 'Ramen', 'Sushi', 'Okonomiyaki', 'Karaage']
nb_classes = len(categories)
# 이미지 크기
image_size = 100

def main():
    data = np.load('./saveFiles/japanese_food_aug.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    Y_train = data['Y_train']
    Y_test = data['Y_test']
    X_train = X_train.astype('float') / 256
    X_test = X_test.astype('float') / 256

    # 레이블 데이터를 원-핫 인코딩으로 변환
    Y_train = utils.to_categorical(Y_train, nb_classes)
    Y_test = utils.to_categorical(Y_test, nb_classes)
    # 모델을 훈련하고 평가하기
    model = model_train(X_train, Y_train)
    model_eval(model, X_test, Y_test)

# CNN 모델 구축
def build_model(in_shape):
    model = Sequential()
    model.add(Conv2D(32, 3, 3, padding='same', input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 은닉층1
    model.add(Conv2D(64, 3, 3, padding='same'))
    model.add(Activation('relu'))

    # 은닉층2 : 세번째 합성곱 층
    model.add(Conv2D(64, 3, 3))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    # 은닉층3 : 완전 연결층(Fully Connected Layer)
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # 출력층
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def model_train(X, Y):
    # 입력 데이터의 shape을 기반으로 모델 생성
    model = build_model(X.shape[1:])
    # 모델 학습 수행
    model.fit(X, Y, batch_size=32, epochs=50)
    # 모델 가중치 저장하기
    hdf5_file = './saveFiles/japanese_food_aug_model.hdf5'
    model.save_weights(hdf5_file)
    return model

# 테스트 데이터로 모델 평가
def model_eval(model, X, Y):
    score = model.evaluate(X, Y)
    print('loss =', score[0])
    print('accuracy =', score[1])

if __name__ == '__main__':
    main()