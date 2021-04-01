import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
from tensorflow.python.compiler.tensorrt import trt_convert as trt

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
time_i = 0
elapsed = 0
time_tot = 0
elapsed_tot = 0
#MODEL IMPORT
saved_model_dir_trt = 'Models/HandsNet_1_trt32'
model = tf.saved_model.load(saved_model_dir_trt)
model.trainable = False 
signature_keys = list(model.signatures.keys())
print('SIG KEYS', signature_keys) # Outputs : ['serving_default']
#Inferece
prediction_model = model.signatures['serving_default']
print('MODEL', prediction_model.structured_outputs)


#HAND IMAGE
image_path= 'dataset/hands/491.png'
img = tf.keras.preprocessing.image.load_img(image_path, target_size=(100, 68))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0) #not sure i need this
#print(type(x))

image_input = tf.constant(x.astype('float32'))#tf.constant Creates a constant tensor from a tensor-like object.
one_prediction = prediction_model(input_1=image_input)
#print(one_prediction)
print(one_prediction['dense_2'].numpy()) #dense_2 is the DICTIONARY VALUE, I am accessing THIS IS MY WAY OF ACCESSING THE DICTIONARY AND GETING THE VALUE

if one_prediction['dense_2'].numpy()[0][0]>0.9:
    print('Hand')
elif one_prediction['dense_2'].numpy()[0][1]>0.9:
    print('Non-Hand')
else:
    print('Not Recognized')


#NONHAND IMAGE
image_path=  'dataset/non_hands/122.png'
img = tf.keras.preprocessing.image.load_img(image_path, target_size=(100, 68))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0) #not sure i need this


start_time = time.time()
image_input = tf.constant(x.astype('float32'))#tf.constant Creates a constant tensor from a tensor-like object.
one_prediction = prediction_model(input_1=image_input)
print(one_prediction['dense_2'].numpy())
delta = (time.time() - start_time)
print("Prediction_Time:", delta)

if one_prediction['dense_2'].numpy()[0][0]>0.9:
    print('Hand')
elif one_prediction['dense_2'].numpy()[0][1]>0.9:
    print('Non-Hand')
else:
    print('Not Recognized')