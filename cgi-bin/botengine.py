# 필요 패키지 임포트
import codecs
import tensorflow as tf
import keras
import numpy as np
import random, sys
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import load_model
from bs4 import BeautifulSoup

#print(tf.__version__)

# 모델 로드
model = load_model('./data/chatbot.model')

# 파일 로드 함수정의
def load_data(file):
  result = []

  with open(file, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

    for line in lines:
      data = line.split(',')
      data = data[0] + ' ' + data[1]
      result.append(data)

  result = result[1:] # header 정보 제외
  return result

# 데이터 로드
chat_dataset = load_data('./data/ChatbotData.csv')
chat_text = ' '.join(chat_dataset)

# 문자 벡터화
chars = sorted(list(set(chat_text)))
char_index = dict((c, i) for i, c in enumerate(chars)) # 문자 - index
index_char = dict((i, c) for i, c in enumerate(chars)) # index - 문자

# 텍스트를 maxlen개의 문자로 자르고 다음에 오는 문자 등록
maxlen = 20
step = 3
sentences = []
next_char = []

for i in range(0, len(chat_text) - maxlen, step):
  sentences.append(chat_text[i: i + maxlen])
  next_char.append(chat_text[i + maxlen])

# 임의의 시작 텍스트 선택
start_index = random.randint(0, len(chat_text) -maxlen - 1)

# 후보 단어를 추천
def sample(pred, temperature=1.0):
    pred = np.asarray(pred).astype('float64')
    pred = np.log(pred) / temperature
    exp_pred = np.exp(pred)

    pred = exp_pred / np.sum(exp_pred)
    probas = np.random.multinomial(1, pred, 1)

    return np.argmax(probas)

def make_reply(sentence):
    generated = ''

    for i in range(40):
        x = np.zeros((1, maxlen, len(chars)))

        for t, char in enumerate(sentence):
            x[0, t, char_index[char]] = 1

        # 다음에 올 문자 예측
        pred = model.predict(x, verbose=0)[0]
        next_index = sample(pred, 0.8)
        next_char = index_char[next_index]

        # 출력
        generated += next_char
        sentence = sentence[1:] + next_char

    return generated