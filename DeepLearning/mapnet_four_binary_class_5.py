#!/usr/bin/python
'''The first model to train a CNN that uses EM density map as input samples 
   and 3-D matrix derived from pdb file
'''


import keras
#from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Conv3D, Reshape, Flatten, MaxPooling3D, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN,ModelCheckpoint, CSVLogger
import keras.utils
from keras import utils as np_utils
from keras import optimizers
from keras import regularizers
from keras import backend as K
import tensorflow as tf
import h5py
import numpy as np
import random
import time

import sys

box_size=sys.argv[1]
maps=sys.argv[2]
ratio=sys.argv[3]
print("box size",sys.argv[1])
print("map size",sys.argv[2])
print("ratio size",sys.argv[3])

print("input argument is working")

############# global variables ##############################################################################
start_time=time.time()
#limit=100
epochs = 300 # how many times we go through each sample in one iteration before we go to the next sample
stop_here_please = EarlyStopping(monitor='val_loss',patience=10)
stop_immediately=TerminateOnNaN()
csv_logger = CSVLogger("logs/training_box_"+str(box_size)+"_maps_"+str(maps)+"_ratio_"+str(ratio)+".log")
save_best_model=ModelCheckpoint('models/best/my_map_model_box_'+str(box_size)+'_maps_'+str(maps)+'_ratio_'+str(ratio)+'_best.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

############# the data preparation ###########################################################################
#loading training data
x_train= np.load('train_data_binary/train_data_box_'+str(box_size)+'_'+str(maps)+'_ratio_'+str(ratio)+'.npy')

print("length of x_train: ",len(x_train))
print("shape of x_train: ",x_train.shape)
print("type of x_train: ",type(x_train))

y_train= np.load('train_data_binary/train_labels_box_'+str(box_size)+'_'+str(maps)+'_ratio_'+str(ratio)+'.npy')

print("length of y_train: ",len(y_train))
print("shape of y_train: ",y_train.shape)
print("type of y_train: ",type(y_train))

# x reshape, x dimension is set to 5 (sample,dim1,dim2,dim3,channel)
print("shape of x_train before: ",x_train.shape)  
x_train = np.expand_dims(x_train, axis=4)
print("shape of x_train after: ",x_train.shape)

# y reshape, y dimension is set to 5 (sample,dim1,dim2,dim3,channel)
print("shape of y_train before: ",y_train.shape)  
#y_train = np.expand_dims(y_train, axis=4)
#print("shape of x_train after: ",y_train.shape)

############# the sequantial model ##############################################################################
model = Sequential()
model.add(Conv3D(10, kernel_size=(3,3,3), activation='relu', input_shape=(5,5,5,1), padding='same'))
model.add(Conv3D(12, kernel_size=(3,3,3), activation='relu', padding='same'))
model.add(Conv3D(24, kernel_size=(3,3,3), activation='relu', padding='valid'))
model.add(Conv3D(48, kernel_size=(3,3,3), activation='relu', padding='same'))
model.add(Conv3D(72, kernel_size=(3,3,3), activation='relu', padding='same'))
model.add(Conv3D(96, kernel_size=(3,3,3), activation='relu', padding='valid'))
#model.add(MaxPooling3D(pool_size=(3,3,3)))
model.add(Flatten())
#model.add(Dense(100,activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(10,activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

nadam_adj = keras.optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
model.compile(loss='binary_crossentropy', optimizer=nadam_adj,metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='nadam',metrics=['accuracy'])

############# model training ##############################################################################
model.fit(x_train, y_train, batch_size=16, epochs=epochs, verbose=2,validation_split=0.2,shuffle=True,callbacks=[csv_logger,save_best_model,stop_immediately,stop_here_please])
model.save('models/last/my_map_model_box_'+str(box_size)+'_maps_'+str(maps)+'_ratio_'+str(ratio)+'_last.h5')  
 
############# model evaluation (training) ##############################################################################  

print(model.summary())
print("--- %s seconds ---" % (time.time() - start_time))





 

