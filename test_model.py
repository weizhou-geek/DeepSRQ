__author__ = 'weizhou'


import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input,Conv2D,ELU,MaxPooling2D,Flatten,Dense,Dropout,normalization
from keras.optimizers import SGD,rmsprop,adam
from keras import backend as K
import os
import scipy.io as sio


#left image
left_image=Input(shape=(32, 32, 3))
#conv1
left_conv1=Conv2D(16, (3, 3), padding='same', name='conv1_left')(left_image)
left_elu1=ELU()(left_conv1)
left_pool1=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1_left')(left_elu1)
#conv2
left_conv2=Conv2D(16, (3, 3), padding='same', name='conv2_left')(left_pool1)
left_elu2=ELU()(left_conv2)
left_pool2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_left')(left_elu2)
#conv3
left_conv3=Conv2D(32, (3, 3), padding='same', name='conv3_left')(left_pool2)
left_elu3=ELU()(left_conv3)
#conv4
left_conv4=Conv2D(32, (3, 3), padding='same', name='conv4_left')(left_elu3)
left_elu4=ELU()(left_conv4)
#conv5
left_conv5=Conv2D(64, (3, 3), padding='same', name='conv5_left')(left_elu4)
left_elu5=ELU()(left_conv5)
left_pool5=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5_left')(left_elu5)
#fc6
left_flat6=Flatten()(left_pool5)
left_fc6=Dense(128)(left_flat6)
left_elu6=ELU()(left_fc6)
left_drop6=Dropout(0.35)(left_elu6)
#fc7
left_fc7=Dense(128)(left_drop6)
left_elu7=ELU()(left_fc7)
left_drop7=Dropout(0.5)(left_elu7)

#right image
right_image=Input(shape=(32, 32, 3))
#conv1
right_conv1=Conv2D(16, (3, 3), padding='same', name='conv1_right')(right_image)
right_elu1=ELU()(right_conv1)
right_pool1=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1_right')(right_elu1)
#conv2
right_conv2=Conv2D(16, (3, 3), padding='same', name='conv2_right')(right_pool1)
right_elu2=ELU()(right_conv2)
right_pool2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_right')(right_elu2)
#conv3
right_conv3=Conv2D(32, (3, 3), padding='same', name='conv3_right')(right_pool2)
right_elu3=ELU()(right_conv3)
#conv4
right_conv4=Conv2D(32, (3, 3), padding='same', name='conv4_right')(right_elu3)
right_elu4=ELU()(right_conv4)
#conv5
right_conv5=Conv2D(64, (3, 3), padding='same', name='conv5_right')(right_elu4)
right_elu5=ELU()(right_conv5)
right_pool5=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5_right')(right_elu5)
#fc6
right_flat6=Flatten()(right_pool5)
right_fc6=Dense(128)(right_flat6)
right_elu6=ELU()(right_fc6)
right_drop6=Dropout(0.35)(right_elu6)
#fc7
right_fc7=Dense(128)(right_drop6)
right_elu7=ELU()(right_fc7)
right_drop7=Dropout(0.5)(right_elu7)

#concatenate
fusion3_drop7=keras.layers.concatenate([left_drop7,right_drop7])
#fc8
fusion3_fc8=Dense(256)(fusion3_drop7)
#fc9
predictions=Dense(1)(fusion3_fc8)

model_all=Model(input=[left_image,right_image],output=predictions,name='all_model')
model_all.summary()


X_testLeft = np.load('./test_image_structure.npy')
X_testRight = np.load('./test_image_lbp.npy')
X_testLeft = X_testLeft.astype('float32')
X_testRight = X_testRight.astype('float32')
X_testLeft /= 255
X_testRight /= 255
X_testLeft -= np.mean(X_testLeft, axis = 0) # zero-center
X_testLeft /= np.std(X_testLeft, axis = 0) # normalize
X_testRight -= np.mean(X_testRight, axis = 0) # zero-center
X_testRight /= np.std(X_testRight, axis = 0) # normalize

model_all.load_weights('./model.hdf5')
y_predict=model_all.predict([X_testLeft,X_testRight], batch_size=10)
sio.savemat('predictScore.mat', {'filename': X_testLeft, 'score': y_predict})
