import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

data = []
labels =[]
classes = 43
cur_path = os.getcwd()

for i in range(classes):
  path = os.path.join(cur_path,'train',str(i))
  images = os.listdir(path)
  # count = 0
  for a in images:

    # if count == 10:
    #   break
    # count+=1

    try:
      image = Image.open(path + '/' + a)
      image = image.resize((30,30))
      image = np.array(image)
      data.append(image)
      labels.append(i)
    except:
      print("Error loading img")
data= np.array(data)
labels = np.array(labels)

print("data",data.shape)
print("labels",labels.shape)
X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=0.2,random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
y_train = to_categorical(y_train,43)
y_test = to_categorical(y_test,43)

'''
  model1:traffic_classifier.h5
  model2:traffic_classifier01.h5
  model3:traffic_classifier02.h5
'''

# model1
# model = Sequential()
# model.add(Conv2D(filters=32,kernel_size=(5,5),activation='relu',input_shape=X_train.shape[1:]))
# model.add(Conv2D(filters=32,kernel_size=(5,5),activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(rate=0.25))
# model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
# model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(rate=0.25))
# model.add(Flatten())
# model.add(Dense(256,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dropout(rate=0.5))
# model.add(Dense(43,activation='softmax'))
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#model 2
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Train

#model1
# epochs=30
# history=model.fit(X_train,y_train,batch_size=64,epochs=epochs,validation_data=(X_test,y_test))

#model2
epochs=30
history=model.fit(X_train,y_train,batch_size=32,epochs=epochs,validation_data=(X_test,y_test))

#model3
# epochs=30
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam
# from pyimagesearch.trafficsignnet import TrafficSignNet

# aug = ImageDataGenerator(
# 	rotation_range=10,
# 	zoom_range=0.15,
# 	width_shift_range=0.1,
# 	height_shift_range=0.1,
# 	shear_range=0.15,
# 	horizontal_flip=False,
# 	vertical_flip=False,
# 	fill_mode="nearest")
# INIT_LR = 1e-3
# opt = Adam(lr=INIT_LR, decay=INIT_LR / (epochs * 0.5))
# model = TrafficSignNet.build(width=32, height=32, depth=3,
# 	classes=len(np.unique(y_train)))
# model.compile(loss="categorical_crossentropy", optimizer=opt,
# 	metrics=["accuracy"])

# history = model.fit_generator(
# 	aug.flow(X_train, y_train, batch_size=32),
# 	validation_data=(X_test, y_test),
# 	steps_per_epoch=X_train.shape[0],
# 	epochs=epochs,
# 	# class_weight=y_train.sum(axis=0).max(),
# 	# verbose=1
#   )

#Mathplotlib
plt.figure(0)
plt.plot(history.history['accuracy'],label='training accuracy')
plt.plot(history.history['val_accuracy'],label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(1)
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'],label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

from sklearn.metrics import accuracy_score
import pandas as pd
y_test = pd.read_csv('Test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data = []

for img in imgs:
  image = Image.open(img)
  image = image.resize((30,30))
  data.append(np.array(image))

X_test=np.array(data)

prev = model.predict_classes(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(labels,prev))
model.save('traffic_classifier02.h5')