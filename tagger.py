
import os, sys, json
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten

def build(vocab, tags):
    model = Sequential()
    model.add(Embedding(len(vocab), 100, input_length=5)) # [None, win_size, embedding_dim]
    model.add(Flatten()) # [None, win_size * embedding_dim]
    model.add(Dense(units=100, input_dim=100 * 5)) # [None, 100]
    model.add(Activation('relu'))
    model.add(Dense(units=len(tags)))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    print model.summary()
    return model

def train(vocab, tags, train_x, train_y, dev_x, dev_y, epochs=5):
    model = build(vocab, tags)
    print train_x.shape
    print train_y.shape
    best_accu = 0.0
    for i in range(epochs):
        model.fit(train_x, train_y, epochs=1, batch_size=50)
        dev_y_hat = model.predict(dev_x, batch_size=50)
        accu = np.mean(np.equal(dev_y.flatten(), np.argmax(dev_y_hat, axis=-1).flatten()))
        print 'epoch', i, 'accuracy', accu
        if best_accu < accu:
            print 'saving model %.4f (prev_best) < %.4f (cur)' % (best_accu, accu)
            best_accu = accu
            model.save('postagger.h5')

def decode(model):
    classes = model.predict(x_test, batch_size=128)

data = json.load(open('data.json','rb'))
vocab = data['vocab']
tags = data['tags']

train_x = np.array(data['train_x'], dtype='int32')
train_y = np.array(data['train_y'], dtype='int32')
print 'training set size:', train_x.shape[0]

dev_x = np.array(data['dev_x'], dtype='int32')
dev_y = np.array(data['dev_y'], dtype='int32')
print 'dev set size:', dev_x.shape[0]

train(vocab, tags, train_x, train_y, dev_x, dev_y)

