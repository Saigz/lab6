import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential
import pathlib

dataset_dir = pathlib.Path("flower_photos")
batch_size = 32
img_width = 320
img_height = 232

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir, 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=11,
    image_size = (img_height, img_width))
class_names = train_ds.class_names

loaded_model = tf.keras.models.load_model('my_model.keras')

img = tf.keras.utils.load_img(
    "test.jpg", target_size=(img_height, img_width)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# делаем прогнозы
predictions = loaded_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# результат вывода 
print("На изображении скорее всего {} ({:.2f}% вероятность)".format(
  class_names[np.argmax(score)],
  100 * np.max(score)))