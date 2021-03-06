# -*- coding: utf-8 -*-
"""Features Wav - Stats.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XlZWtWZrKxDlDdaCV69fzfJ2ZPU4y6oD
"""

# -*- coding: utf-8 -*-
"""
This takes the data for one ppt in one session and organises it into
two arrays, one for left and one for right motor imagery data. 
All data not related to the MI trials is disregarded.
"""
import numpy as np
import glob
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.activations import sigmoid, relu
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


import scipy.io as spio
import matplotlib.pyplot as plt

def figure(History, legend):
  ####################### PLOT TRAINING VS VALIDATION ######################
  ########## Accuracy ###########
  acc = History.history['acc']
  val_acc = History.history['val_acc']
  loss = History.history['loss']
  val_loss = History.history['val_loss']

  plt.plot(acc)
  plt.plot(val_acc)
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(legend, loc='upper left')
  plt.grid()
  plt.show()

  ########## Loss ###########
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  try:
    loss_no_reg = History.history['categorical_crossentropy']
    val_loss_no_reg = History.history['val_categorical_crossentropy']
    plt.plot(loss_no_reg)
    plt.plot(val_loss_no_reg)
    plt.legend(legend + [legend[0]+' sin reg', legend[1]+' sin reg'], loc='upper left')
  except:
    plt.legend(legend, loc='upper left')

  plt.grid()
  plt.show()

from google.colab import files
uploaded = files.upload()

# #change path
path_TRAIN = r'/content/P3_post_test_caract15s.mat' # use your path
path_test = r'/content/P3_post_training_caract15s.mat' # use your path
#path = r'/content'
#files_TRAIN = glob.glob(path + "/*training.mat")
#files_test = glob.glob(path + "/*test.mat")

mat_TRAIN = spio.loadmat(path_TRAIN, squeeze_me=True)
mat_test = spio.loadmat(path_test, squeeze_me=True)

mat_TRAIN['mcaract'].shape

y_TRAIN_orig = np.clip( np.max(mat_TRAIN['vlabel'].reshape([-1,4]), axis=1) , 0 , 1)
x_TRAIN_orig = mat_TRAIN['mcaract'].reshape([-1,4,mat_TRAIN['mcaract'].shape[1]])
y_test_orig = np.clip( np.max(mat_test['vlabel'].reshape([-1,4]), axis=1) , 0 , 1)
x_test_orig = mat_test['mcaract'].reshape([-1,4,mat_TRAIN['mcaract'].shape[1]])

y_TRAIN_orig

x_TRAIN_orig.shape

max_value = np.max(x_TRAIN_orig, axis=(0,1))
min_value = np.min(x_TRAIN_orig, axis=(0,1))

x_TRAIN = (x_TRAIN_orig-min_value)/(max_value-min_value)
x_test = (x_test_orig-min_value)/(max_value-min_value)

print(x_TRAIN.shape)
print(x_test.shape)
print(y_TRAIN_orig.shape)
print(y_test_orig.shape)

model = tf.keras.models.Sequential([
  #tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(x_TRAIN.shape[1],x_TRAIN.shape[2])),
  tf.keras.layers.LSTM(8, input_shape=(x_TRAIN.shape[1],x_TRAIN.shape[2])),
  #tf.keras.layers.LSTM(8),
  tf.keras.layers.Dense(1, activation = sigmoid),
])
model.summary()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-6 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.Adam(lr=1e-6)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=["acc"])
history = model.fit(x_TRAIN, y_TRAIN_orig, epochs=100, callbacks=[lr_schedule])

plt.semilogx(history.history["lr"], history.history["loss"])
#plt.axis([1e-8, 1e-4, 0, 30])

model = tf.keras.models.Sequential([
  #tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(x_TRAIN.shape[1],x_TRAIN.shape[2])),
  #tf.keras.layers.LSTM(32),
  tf.keras.layers.LSTM(8, input_shape=(x_TRAIN.shape[1],x_TRAIN.shape[2]) ),
  tf.keras.layers.Dense(1, activation = sigmoid),
])

model.summary()

optimizer = tf.keras.optimizers.Adam(lr=2e-3)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=["acc"])
# Early stopping
stopping = EarlyStopping(monitor='val_loss',min_delta=0,mode='auto',patience=50, restore_best_weights=True)

history = model.fit(x_TRAIN, y_TRAIN_orig, callbacks=[stopping], validation_data = (x_test, y_test_orig), epochs=200)

figure(history, ['train', 'test'])

plt.plot(model.predict_classes(x_test))
plt.plot(y_test_orig)
model.evaluate(x_test, y_test_orig)

