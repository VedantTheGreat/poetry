import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Activation
from tensorflow.keras.optimizers import RMSprop
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath,'rb').read().decode('utf-8').lower()
text = text[300000:800000]
char = sorted(set(text))
char_to_index = dict((c, i) for i, c in enumerate(char))
index_to_char = dict((i,c) for i, c in enumerate(char))
SEQ_LENGTH = 40
STEP_SIZE = 3
sent = []
next_char = []
for i in range(0, len(text)- SEQ_LENGTH, STEP_SIZE):
    sent.append(text[i:i + SEQ_LENGTH])
    next_char.append(text[i + SEQ_LENGTH])
x = np.zeros((len(sent),len(char),  SEQ_LENGTH), dtype=bool)
y = np.zeros((len(sent), len(char)), dtype=bool)
for i, sentence in enumerate(sent):
    for j, char in enumerate(sentence):
        x[i,char_to_index[char], j] = 1
    y[i,char_to_index[next_char[i]]] = 1
model = Sequential()
print(x.shape)
print(y.shape)
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(char))))
model.add(Dense(len(char)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer = RMSprop(learning_rate = 0.01))
model.fit(x,y,batch_size = 256, epochs=4)
model.save('gen.model')