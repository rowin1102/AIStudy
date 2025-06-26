from PIL import Image, ImageEnhance
import glob
import numpy as np
from sklearn.model_selection import train_test_split

root_dir = './download'
categories = ['Gyudon', 'Ramen', 'Sushi', 'Okonomiyaki', 'Karaage']
nb_classes = len(categories)
image_size = 100

X = []
Y = []

# 이미지 데이터를 불러와서 리스트에 추가
def add_sample(cat, fname, is_train):
    img = Image.open(fname)
    img = img.convert('RGB')
    img = img.resize((image_size, image_size))
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)

    # 테스트 데이터인 경우 데이터 증강을 하지 않음
    if not is_train: return
    # 데이터 증강
    for ang in range(-30, 30, 10):
        # 이미지 회전 (-30~30 사이에서 적용됨)
        img2 = img.rotate(ang)
        # 넘파이 배열로 만든 후 리스트에 추가
        data = np.asarray(img2)
        X.append(data)
        Y.append(cat)
        # img2.save('jfood-' + str(ang) + '.png')

        # 좌우 반전(Flip) 적용

        # ImageEnhance 모듈을 이용해서 아래의 증강 작업을 할 수 있음
        # 밝기 조절 (랜덤값 적용) : Brightness()
        # 대비 조절 (랜덤값 적용): Contrast()

# 데이터를 생성하는 함수
def make_sample(files, is_train):
    global X, Y
    X = []; Y = []
    for cat, fname in files:
        add_sample(cat, fname, is_train)
    return np.array(X), np.array(Y)

# 각 폴데에 들어있는 파일 수집하기
allfiles = []

for idx, cat in enumerate(categories):
    image_dir = root_dir + '/' + cat
    files = glob.glob(image_dir + '/*.jpg')
    print('---', cat, '처리 중')
    # 리스트에 파일명을 추가
    for f in files:
        print(f)
        allfiles.append((idx, f))

# 섞은 뒤에 학습 전용 데이터와 테스트 전용 데이터 구분
# random.shuffle(allfiles)

# 클래스별 균등한 데이터 분할 적용
train, test = train_test_split(allfiles, test_size=0.3, stratify=[x[0] for x in allfiles],
                               random_state=42)

# 훈련용, 테스트용 데이터 모두 증강시켜 리스트에 추가한다.
X_train, Y_train = make_sample(train, True)
X_test, Y_test = make_sample(test, False)

# Numpy 배열 저장
np.savez('./saveFiles/japanese_food_aug.npz', X_train=X_train, X_test=X_test,
         Y_train=Y_train, Y_test=Y_test)
print('Task Finished..!!', len(Y))