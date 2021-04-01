import os
import numpy as np
import tensorflow as tf
from sympy import print_ccode
from tensorflow import keras

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
image_path_hands = 'dataset/hands/491.png'
image_path_non_hands = 'dataset/non_hands/122.png'

#IMPORTING THE MODEL
new_HandsNet = tf.keras.models.load_model('Models/HandsNet_2')
#new_HandsNet.summary()
new_HandsNet.trainable=False


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    validation_split=0.3,
    subset="validation",
    labels="inferred",
    seed = 71810,
    image_size=image_size,
    batch_size=batch_size,
)
# Evaluate the restored model
scores = new_HandsNet.evaluate(val_ds, verbose=1, return_dict=True)
print('Restored model')
print(scores)
print(new_HandsNet.predict(val_ds).shape)

#Confusion_Matrix
y_pred=new_HandsNet.predict_classes(val_ds)
y_true=np.zeros(1059)
con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)



image_hand = tf.keras.preprocessing.image.load_img(image_path_hands, target_size = image_size)
input_arr_hand= keras.preprocessing.image.img_to_array(image_hand)
input_arr_hand = np.array([input_arr_hand])  # Convert single image to a batch.
predictions = new_HandsNet.predict(input_arr_hand)
print(predictions)

image_non_hand = tf.keras.preprocessing.image.load_img(image_path_non_hands, target_size = image_size)
print(type(image_non_hand))

input_arr_non_hand= keras.preprocessing.image.img_to_array(image_non_hand)
print(type(input_arr_non_hand))
print(input_arr_non_hand.shape)

input_arr_non_hand = np.array([input_arr_non_hand])  # Convert single image to a batch.
print(type(input_arr_non_hand))
print(input_arr_non_hand.shape)
predictions = new_HandsNet.predict(input_arr_non_hand)
print(predictions)

