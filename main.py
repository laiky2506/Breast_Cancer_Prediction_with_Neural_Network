# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 12:16:23 2022

@author: laiky
"""

import numpy as np
import tensorflow as tf
import pandas as pd

#load csv data into pd dataframe
data = pd.read_csv('data.csv')

#set column id become index
data.set_index('id')

#check data completeness
print(data.info())

#from observation, the data is very organize without any missing data, we only need to remove col 32 Unnamed: 32
data = data.drop(labels=['Unnamed: 32'], axis=1)

#%%
#set labels and features
label = data['diagnosis']
label = pd.get_dummies(label, drop_first=True, prefix='diagnosis')

#features: drop id and diagnosis
features = data.drop(labels=['diagnosis','id'], axis=1)

#%%
#Train Test Split and Standardize data
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
SEED = 12345
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=SEED)
x_train = np.array(x_train)
x_test = np.array(x_test)

standardizer = preprocessing.StandardScaler()
standardizer.fit(x_train)
x_train = standardizer.transform(x_train)
x_test = standardizer.transform(x_test)

#%%
#Setting of the feed forward neural network

nClass = len(np.unique(y_test))
inputs = tf.keras.Input(shape=(x_train.shape[-1],))
dense = tf.keras.layers.Dense(64,activation='relu')
x = dense(inputs)
dense = tf.keras.layers.Dense(32,activation='relu')
x = dense(x)
dense = tf.keras.layers.Dense(16,activation='relu')
x = dense(x)
outputs = tf.keras.layers.Dense(nClass,activation='softmax')(x)
model = tf.keras.Model(inputs=inputs,outputs=outputs,name='breast_cancer_model')
model.summary()

#%%
#start the iteration

callback = tf.keras.callbacks.EarlyStopping(patience=5)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=32,epochs=128,callbacks=[callback])

#%%
#Plot model accuracy graph

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()