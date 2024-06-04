import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import re
import shutil
import string
import tensorflow as tf

from keras import layers
from keras import losses

url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
dataset = tf.keras.utils.get_file("stack_overflow_16k", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'train')