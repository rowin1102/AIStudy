import tensorflow as tf
import pandas as pd
import numpy as np
import datetime

csv = pd.read_csv('./resData/bmi.csv')

csv['height'] = csv['height'] / 200
csv['weight'] = csv['weight'] / 100

bclass = {'thin': [1, 0, 0], 'normal': [0, 1, 0], 'fat': [0, 0, 1]}
csv['label_pat'] = csv['label'].apply(lambda x: np.array(bclass[x]))
print(csv.head())

test_csv = csv[15000:20000]
test_pat = np.array(test_csv[['weight', 'height']])
test_ans = np.array(list(test_csv['label_pat']))

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

train_pat = np.array(csv[['weight', 'height']])
train_ans = np.array(list(csv['label_pat']))

# 로그 디렉터리 설정 (현재 시간 기반 폴더 생성)
log_dir = 'log_dir/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)

history = model.fit(
    train_pat, train_ans,
    epochs = 35,
    batch_size = 100,
    validation_data = (test_pat, test_ans),
    verbose = 1,
    callbacks = [tensorboard_callback],
)

test_loss, test_acc = model.evaluate(test_pat, test_ans)
print('정답룰 =', test_acc)

with tf.summary.create_file_writer(log_dir).as_default():
    tf.summary.scalar('Test Accuracy', test_acc, step=0)
    tf.summary.scalar('Test Loss', test_loss, step=0)

print(f"TensorBoard write ok : {log_dir}")