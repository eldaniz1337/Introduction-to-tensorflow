#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def conv_classifier():

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
         def on_epoch_end(self, callback, logs={}):
                if (logs.get('acc') > DESIRED_ACCURACY):
                    print("\nReached {}/% accuracy, stopping training".format(DESIRED_ACCURACY*100))
                    self.model.stop_training = True

    callbacks = myCallback()
    
    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([
        # Your Code Here
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])
        

    # This code block should create an instance of an ImageDataGenerator called train_datagen 
    # And a train_generator by calling train_datagen.flow_from_directory

    train_datagen = ImageDataGenerator(rescale=1./255)

    # target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory(
        '/train/',
        target_size=(150, 150),
        batch_size=10,
        class_mode='binary'
    )

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=8,
        epochs=15,
        verbose=1,
        callbacks=[callbacks]
    )
    # model fitting
    return history.history['acc'][-1]


print(conv_classifier())
