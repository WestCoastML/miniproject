from __future__ import print_function
import os
os.environ['THEANO_FLAGS'] = "device=gpu0,floatX=float32"
os.environ['PATH'] = '/usr/local/cuda/bin:{}'.format(os.environ['PATH'])
import theano

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.engine import InputSpec
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten,Merge,  Input,Layer,merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from attention import SpatialTransformer
#from attention import SpatialTransformerLayer

batch_size = 100
nb_classes = 9

nb_epoch = 50
nb_filters=32
kernel_size=(3,3)
pool_size=(2,2)

#image_size=(218,303)
#input_shape=(3,218,303)

#image_shape = (None, 128, 128, 1)
#images = Input(shape=image_shape[1:])

image_size=(128,128)
input_shape=(3,128,128)
classes=["chicken","ostrich",'bluebird','finch','frog','salemander','cobra','bird','flamingo']

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        '../data/train',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical')
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        '../data/val',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical')


input_shape=(3,128,128)
img = Input(shape=input_shape)

# initial weights
b = np.zeros((2, 3), dtype='float32')
b[0, 0] = 1
b[1, 1] = 1
W = np.zeros((50, 6), dtype='float32')
weights = [W, b.flatten()]

modela = Sequential()
modela.add(MaxPooling2D(pool_size=(2,2), input_shape=input_shape))
modela.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
modela.add(Activation('relu'))
modela.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
modela.add(Activation('relu'))
modela.add(Flatten())
modela.add(Dense(50))
modela.add(Activation('relu'))
modela.add(Dense(6,weights=weights))

modelb = Sequential()
modelb.add(SpatialTransformer(modela,input_shape=input_shape))
modelb.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
modelb.add(Activation('relu'))
modelb.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
modelb.add(Activation('relu'))
modelb.add(Flatten())
modelb.add(Dense(9))
modelb.add(Activation('softmax'))

model = Model(input=img, output=modelb(img))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
model.summary()
model.fit_generator(train_generator,samples_per_epoch=1000, nb_epoch=nb_epoch)
score = model.evaluate_generator(train_generator, val_samples=10)
print('Test score:', score[0])
print('Test accuracy:', score[1])
