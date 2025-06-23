import random

# BMI를 계산해서 레이블을 리턴하는 함수
def calc_bmi(h, w):
    bmi = w / (h / 100) ** 2
    # 지수에 따른 판단
    if bmi < 18.5: return  'thin'
    if bmi < 25: return 'normal'
    return 'fat'

# 출력 파일 준비하기
fp = open('./resData/bmi.csv', 'w', encoding='utf-8')
# 쓰기 모드로 오픈한 후 첫 번째 줄에는 컬럼명을 입력
fp.write('height,weight,label\r\n')

# 무작위로 데이터 생성하기
cnt = {'thin':0, 'normal':0, 'fat':0}

# 2만개의 데이터 반복
for i in range(20000):
    h = random.randint(120, 200)
    w = random.randint(35, 80)
    label = calc_bmi(h, w)
    cnt[label] += 1
    fp.write('{0},{1},{2}\r\n'.format(h, w, label))
# 모든 입력이 완료되면 파일 객체를 닫아준다.
fp.close()
print('ok', cnt)