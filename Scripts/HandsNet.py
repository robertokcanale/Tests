import matplotlib.pyplot as plt
import os
import numpy as np
from numpy.testing._private.utils import print_assert_equal
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow import keras
from tensorflow.keras import callbacks, layers
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.core import Activation
#checkpoint.path


#LIMITING THE GPU, i need this or it blows away
gpus = tf.config.experimental.list_physical_devices('GPU')
#limit exponential growth
tf.config.experimental.set_memory_growth(gpus[0], True)
if gpus:
    # Restrict TensorFlow to only allocate 4GB  of memory on the first GPU
    try:
       #setting limit
       tf.config.experimental.set_virtual_device_configuration(gpus[0],
           [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 4)])
       logical_gpus = tf.config.experimental.list_logical_devices('GPU')
       print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

#HYPERPARAMETERS
seed = 9682
epochs= 130 #Albini uses 80
batch_size = 64 
starter_learning_rate = 0.1
end_learning_rate = 0.02
decay_steps = 100000
boundaries = [1, 1000]
values = [0.05, 0.005, 0.0005]
#PolynomialDecay
#learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(starter_learning_rate, decay_steps, end_learning_rate, power=0.5)
#PiecewiseConstDecay
#learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

#the preprocessinge image dataset from directory can easily reshape to the required size
image_size = (100, 68)

#DATASET GENERATION

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    validation_split=0.3,
    subset="training",
    labels="inferred",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    validation_split=0.3,
    subset="validation",
    labels="inferred",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
)

#STUFF I STILL DO NOT UNDERSTAND
#use disk data without IO becoming blocking
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#PREPROCESSING
data_augmentation = keras.Sequential(
  [  
    layers.experimental.preprocessing.RandomFlip("horizontal",  seed=seed),      
    layers.experimental.preprocessing.RandomFlip("vertical", seed=seed),    
    #layers.experimental.preprocessing.RandomRotation(0.25 , seed=seed),
    #layers.experimental.preprocessing.RandomRotation((-0.4, -0.1), seed=seed),
    layers.experimental.preprocessing.Rescaling(1./255)

  ]
)


#MODEL
HandsNet = Sequential([
  keras.Input(shape=(1, 100,  68, 3)),
  #preprocessing in the model, although i should actually do it before
  data_augmentation,

  #ConvLayer 1
  layers.Conv2D(32, 7, strides=1, padding='same'),
  layers.BatchNormalization(),
  layers.Activation('relu'),
  layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
  layers.Dropout(0.1),

  #ConvLayer 2
  layers.Conv2D(64, 5, strides=1, padding='same'),
  layers.BatchNormalization(),
  layers.Activation('relu'),
  layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
  layers.Dropout(0.1),

  #ConvLayer 3
  layers.Conv2D(128, 3, strides=1, padding='same'),
  layers.BatchNormalization(),
  layers.Activation('relu'),
  layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),

  #ConvLayer 4
  layers.Conv2D(256, 1, strides=1, padding='same'),
  layers.BatchNormalization(),
  layers.Activation('relu'),
  layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),

  #FCLayer1
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dropout(0.6),

  #FCLayer2
  layers.Flatten(),
  layers.Dense(32, activation='relu'),
  layers.Dropout(0.5),

  #FCLayer3
  layers.Flatten(),
  layers.Dense(2, activation='softmax'),
])

#SETTING MODEL OPTIMIZER
#doing a gradient descent with momentum optimizer, this is a pretty standard and optimized situation
#opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, decay=0.05)
#opt = keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
opt = tf.keras.optimizers.Adam(learning_rate = 0.01)
#TRAIN1
#MODEL COMPILE
HandsNet.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # "binary_crossentropy"
              metrics=['accuracy'])
history = HandsNet.fit(train_ds, epochs=epochs, validation_data=val_ds, batch_size=batch_size)

#TRAIN2 FINE TUNING
opt = tf.keras.optimizers.Adam(learning_rate = 0.0005)
#MODEL COMPILE
HandsNet.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # "binary_crossentropy"
              metrics=['accuracy'])
history = HandsNet.fit(train_ds, epochs=30, validation_data=val_ds, batch_size=batch_size)

HandsNet.save('Models/HN_new/HandsNet_1') # save_format='tf', overwrite=True
HandsNet.save('Models/HN_new/HandsNet_1.h5') 
scores = HandsNet.evaluate(val_ds, verbose=0, return_dict=True)

print(scores)

#VISUALIZE TRAINING RESULS
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
