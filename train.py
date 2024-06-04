import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential
import pathlib

tf.test.is_gpu_available

dataset_dir = pathlib.Path("flower_photos")
batch_size = 32
img_width = 400
img_height = 400

#разбиваем фотки 80:20(80 - тренировка, 20 - валидация)
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir, 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=11,
    image_size = (img_height, img_width))
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir, 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=11,
    image_size = (img_height, img_width))

#выводим все классы изображений
class_names = train_ds.class_names
print(f"Class names: {class_names}")

#кэшируем данные
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#создаем модель
num_classes = len(class_names)
model = Sequential([
    layers.Resizing(img_width, img_height),
    layers.Rescaling(1./255),

    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
  layers.RandomContrast(0.2),

  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

#компилируем её
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#обучение модели
epochs = 5
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

model.save('my_model.keras')

#построим графики
history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs_range = range(1, len(acc) + 1)

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
plt.savefig('graph.png')
