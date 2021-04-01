import os
import numpy as np
import tensorflow as tf
from sympy import print_ccode
from tensorflow import keras
import time
from random import randint
import matplotlib.pyplot as plt

#LIMITING THE GPU, i need this or it blows away
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 5GB  of memory on the first GPU
    try:
       tf.config.experimental.set_virtual_device_configuration(
           gpus[0],
           [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 4)])
       logical_gpus = tf.config.experimental.list_logical_devices('GPU')
       print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

#SETTING UP
image_size = (100, 68)
batch_size = 64 
iterations = 300
time_i = 0
elapsed = np.zeros(iterations, dtype=np.float32)

#KERAS MODEL
new_HandsNet = tf.keras.models.load_model('Models/HandsNet_1')
new_HandsNet.trainable = False 

#TFTRT IMPORT
saved_model_dir_trt = 'Models/HandsNet_1_trt32'
model = tf.saved_model.load(saved_model_dir_trt)
model.trainable = False 
signature_keys = list(model.signatures.keys())
prediction_model = model.signatures['serving_default']

for i in range(iterations):
        im_name='data_prediction/'+str(randint(1, 1700))+'.png'
        image_hand = tf.keras.preprocessing.image.load_img(im_name, target_size = image_size)
        input_arr_hand= keras.preprocessing.image.img_to_array(image_hand)
        input_arr_hand = np.array([input_arr_hand])  # Convert single image to a batch.
        predictions = new_HandsNet.predict(input_arr_hand)

        if predictions[0,0]>0.9:
            print(predictions[0,0])
        elif predictions[0,1]>0.9:
            print(predictions[0,1])
        else: 
            print("KERAS Not Recognized")
        x = tf.keras.preprocessing.image.img_to_array(image_hand)
        x = np.expand_dims(x, axis=0) #not sure i need this

        image_input = tf.constant(x.astype('float32'))#tf.constant Creates a constant tensor from a tensor-like object.
        one_prediction = prediction_model(input_1=image_input)
        if one_prediction['dense_2'].numpy()[0][0]>0.9:
            print(one_prediction['dense_2'].numpy()[0][0])
        elif one_prediction['dense_2'].numpy()[0][1]>0.9:
            print(one_prediction['dense_2'].numpy()[0][1])
        else:
            print('TFTRT Not Recognized')

        
        #print("Prediction time:", elapsed)





