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
chat_dataset = load_data('../data/ChatbotData.csv')
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

train_x=np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
train_y=np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentences in enumerate(sentences):

  for t, char in enumerate(sentences):
    train_x[i, t, char_index[char]] = 1

  train_y[i, char_index[next_char[i]]] = 1

# 모델 구성
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.01))

# 모델 학습
model.fit(train_x, train_y, batch_size=128, epochs=10)

# 모델 저장
model.save('../data/chatbot.model')