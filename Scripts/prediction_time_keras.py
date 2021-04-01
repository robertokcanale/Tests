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
       tf.config.experimental.set_virtual_device_configuration(gpus[0],
           [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 4)])
       logical_gpus = tf.config.experimental.list_logical_devices('GPU')
       print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

#SETTING UP
image_size = (100, 68)
batch_size = 64 
iterations = 500
time_i = 0
elapsed = np.zeros(iterations, dtype=np.float32)

#IMPORTING THE MODEL
new_HandsNet = tf.keras.models.load_model('Models/HandsNet_2')
new_HandsNet.trainable = False 
#new_HandsNet.summary()

for i in range(iterations):
        im_name='data_prediction/'+str(randint(1, 1700))+'.png'
        image_hand = tf.keras.preprocessing.image.load_img(im_name, target_size = image_size)
        time_i = time.time()
        input_arr_hand= keras.preprocessing.image.img_to_array(image_hand)
        input_arr_hand = np.array([input_arr_hand])  # Convert single image to a batch.
        predictions = new_HandsNet.predict(input_arr_hand)
        elapsed[i] = time.time() - time_i

        if elapsed[i] > 0.05: #I do this check because the first iteration also opens the engines or the network, so OUTLAYER cleaning
            elapsed[i] = 0.025
        
        #print("Prediction time:", elapsed)


print("Mean:", np.mean(elapsed), "Median:", np.median(elapsed), "Standard Deviation:", np.std(elapsed))

np.savetxt("test_HN2_keras.txt", elapsed)
#Plotting Results
x=range(iterations)
y=elapsed

plt.figure(figsize=(8, 8))
plt.plot(x, y, label='HN Processing Time')
plt.xlabel("Iteration")
plt.ylabel("Seconds")
plt.title('HN2 Processing Time /per Image')

plt.show()

