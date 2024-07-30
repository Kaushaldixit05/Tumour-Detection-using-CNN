path = "C:/Users/kaushal/OneDrive/Desktop/MINI moye moye project"
import splitfolders

splitfolders.fixed("C:/Users/kaushal/OneDrive/Desktop/MINI moye moye project", output=path+"/output",
    seed=1337, fixed=(260, 260), oversample=False, group_prefix=None, move=False)

import os

for dir, dirname, files in os.walk(path+'/output'):
    print(f"Dir: {dir} | subdir: {dirname} | cant de imagenes: {len(files)} ")

classes = [f for f in os.listdir(path+'/output/train')]
print(classes)

import matplotlib.pyplot as plt
import random

plt.figure(figsize=(10,7))
id = random.choice(range(0,1499))
plt.subplot(1,2,1)
img_name = [f for f in os.listdir(path+'/output/train/Healthy/')][id]
img = plt.imread(path+'/output/train/Healthy/'+img_name)
plt.imshow(img)
plt.title('Healthy '+str(img.shape))
plt.subplot(1,2,2)
img_name = [f for f in os.listdir(path+'/output/train/Brain Tumor/')][id]
img = plt.imread(path+'/output/train/Brain Tumor/'+img_name)
plt.imshow(img)
plt.title('Brain Tumor '+str(img.shape))

lista_img_tumor = [f for f in os.listdir(path+'/output/train/Brain Tumor/')]
imagen_prime = plt.imread(path+'/output/train/Brain Tumor/'+lista_img_tumor[23])
print(f"Shape of the image: {imagen_prime.shape}")
print(f"Dimensiones of the image: {imagen_prime.ndim}")
print(f"Codify the image: {imagen_prime.dtype}")
print(f"Pixel of major value: {imagen_prime.max()}")
print(f"Pixel of minor value: {imagen_prime.min()}")

import cv2
import os

folders = [path+'/output/train/Brain Tumor', path+'/output/train/Healthy', path+'/output/val/Brain Tumor', path+'/output/val/Healthy', path+'/output/test/Brain Tumor', path+'/output/test/Healthy']

for folder in folders:
    files = os.listdir(folder)

    for file in files:
        img = cv2.imread(os.path.join(folder, file))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(folder, file), img_gray)

lista_img_tumor = [f for f in os.listdir(path+'/output/train/Brain Tumor/')]
imagen_prime = plt.imread(path+'/output/train/Brain Tumor/'+lista_img_tumor[23])
print(f"Shape of the image: {imagen_prime.shape}")
print(f"Dimensiones of the image: {imagen_prime.ndim}")
print(f"Codify the image: {imagen_prime.dtype}")
print(f"Pixel of major value: {imagen_prime.max()}")
print(f"Pixel of minor value: {imagen_prime.min()}")

import tensorflow as tf
import keras
tf.random.set_seed(42)

train_dir = path+'/output/train'
test_dir = path+'/output/test'
val_dir = path+'/output/val'

train_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                 horizontal_flip=True,
                                                                 vertical_flip=True)

test_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                horizontal_flip=True,
                                                                vertical_flip=True)

val_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                horizontal_flip=True,
                                                                vertical_flip=True)

train_batch = train_generator.flow_from_directory(train_dir,
                                                 target_size=(224,224),
                                                 batch_size=32,
                                                 color_mode='grayscale',
                                                 class_mode='binary')

test_batch = test_generator.flow_from_directory(test_dir,
                                               target_size=(224,224),
                                               batch_size=32,
                                               color_mode='grayscale',
                                               class_mode='binary')

val_batch = val_generator.flow_from_directory(val_dir,
                                              target_size=(224,224),
                                              batch_size=32,
                                              color_mode='grayscale',
                                              class_mode='binary')


model_0 = keras.Sequential([
    keras.layers.Input(shape=([224,224,1])),
    keras.layers.Conv2D(7, kernel_size=(5,5), activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Dropout((0.01)),
    keras.layers.Conv2D(7, kernel_size=(5,5), activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(7, kernel_size=(5,5), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model_0.summary()


model_0.compile(optimizer = keras.optimizers.Adam(learning_rate=0.00015),
                loss = keras.losses.BinaryCrossentropy(),
                metrics = ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

history = model_0.fit(train_batch,
            epochs=100,
            validation_data = val_batch,
            callbacks = [keras.callbacks.ModelCheckpoint(path+'/checkpoints/model_0', save_best_only=True),
                         keras.callbacks.TensorBoard(path+'/logs/model_0'),
                         keras.callbacks.EarlyStopping(patience=10)])

model_0 = keras.models.load_model(path+'/checkpoints/model_0')
