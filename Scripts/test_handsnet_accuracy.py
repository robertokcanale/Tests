import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from random import randint

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

#SETTING
image_size = (100, 68)
batch_size = 64 
iterations = 500
accuracy= np.zeros(iterations, dtype=np.float32)

#IMPORTING THE MODEL
new_HandsNet = tf.keras.models.load_model('Models/HandsNet_2')
new_HandsNet.summary()
new_HandsNet.trainable=False

#LOOP, here i validate and store the resukt
for i in range(iterations):
    seed= randint(0, 10000)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset",
        validation_split=0.3,
        subset="validation",
        labels="inferred",
        seed = seed,
        image_size=image_size,
        batch_size=batch_size,
    )
    # Evaluate the restored model
    scores = new_HandsNet.evaluate(val_ds, verbose=1, return_dict=True)
    accuracy[i] = scores['accuracy']

print("Mean:", np.mean(accuracy), "Median:", np.median(accuracy), "Standard Deviation:", np.std(accuracy))

#Plotting Results
x=range(iterations)
y=accuracy*100
plt.figure(figsize=(8, 8))
plt.plot(x, y, label='Validation Accuracy')
plt.xlabel("Iteration")
plt.ylabel("%")
plt.title('Validation Accuracy')

plt.show()

