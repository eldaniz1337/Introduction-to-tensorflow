#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
from tensorflow import keras

def house_model(y_new):
    xs = np.array([1,2,3,4,5], dtype=float)
    ys = np.array([1,1.5,2,2.5,3], dtype=float)
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=1500)
    return model.predict(y_new)[0]



prediction = house_model([7.0])
print("{0:.0} hundreds of thousands".format(round(prediction[0])))

print("{:.0f} hundreds of thousands".format(round(prediction[0])))

