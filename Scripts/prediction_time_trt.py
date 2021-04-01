import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
from tensorflow.python.compiler.tensorrt import trt_convert as trt
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

image_size = (100, 68)
batch_size = 64 
iterations = 500
time_i =0
elapsed= np.zeros(iterations, dtype=np.float32)
#MODEL IMPORT
saved_model_dir_trt = 'Models/HandsNet_2_trt32'
model = tf.saved_model.load(saved_model_dir_trt)
model.trainable = False 
signature_keys = list(model.signatures.keys())
#Inferece
prediction_model = model.signatures['serving_default']

for i in range(iterations):
        im_name='data_prediction/'+str(randint(1, 1700))+'.png'
        image_hand = tf.keras.preprocessing.image.load_img(im_name, target_size = (100, 68))
        time_i = time.time()
        input_arr_hand= keras.preprocessing.image.img_to_array(image_hand)
        input_arr_hand = np.expand_dims(input_arr_hand, axis=0)
        image_input = tf.constant(input_arr_hand.astype('float32'))
        one_prediction = prediction_model(input_1=image_input)
        elapsed[i] = time.time() - time_i

        if elapsed[i] > 1: #I do this check because the first iteration also opens the engines or the network, so OUTLAYER cleaning
            elapsed[i] = 0.001
        
        #print("Prediction time:", elapsed)


print("Mean:", np.mean(elapsed), "Median:", np.median(elapsed), "Standard Deviation:", np.std(elapsed))
np.savetxt("test_HN2_32.txt", elapsed)

#Plotting Results
x=range(iterations)
y=elapsed

plt.figure(figsize=(8, 8))
plt.plot(x, y)
plt.xlabel("Image")
plt.ylabel("Seconds")
plt.title('HN2 fp32 Processing Time /per Image')

plt.show()