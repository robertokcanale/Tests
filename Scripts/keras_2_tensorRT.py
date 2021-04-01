import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.python.compiler.tensorrt import trt_convert as trt


save_pb_dir = 'Models/HandsNet_1_trt32'
model_fname = 'Models/HandsNet_1'

params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
      precision_mode='FP32', #trt.TrtPrecisionMode.FP16
      maximum_cached_engines =100,
      is_dynamic_op= True
      #minimum_segment_size=3, is default
      ) 


converter = trt.TrtGraphConverterV2(input_saved_model_dir=model_fname,  conversion_params=params)
converter.convert()

converter.save(save_pb_dir)
