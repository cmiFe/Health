from skimage.io import imread
import pandas as pd
import os
import keras
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import numpy as np
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.models import model_from_json
from keras import regularizers
from keras.regularizers import l1
from matplotlib import pyplot

def classificaValor(lista):
    for i in lista:
        if i < 0.3:
            print()
def modelo(n1,n2,n3):
    model = Sequential()
    model.add(Conv2D(n1, kernel_size=(5, 5), activation='relu',input_shape=(50,50,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(n2, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(n3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])
    history = model.fit(x_treino, y_treino, validation_data=(x_teste, y_teste), epochs=30,batch_size=10)
    return  model,history

def modelo2(n1,n2,n3,n4):
    model = Sequential()
    model.add(Conv2D(n1, kernel_size=(5, 5), activation='relu',input_shape=(50,50,1)))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(n2, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(n3, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(n4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.SGD(lr=0.001),metrics=['accuracy'])
    history = model.fit(x_treino, y_treino, validation_data=(x_teste, y_teste), epochs=30,batch_size=10)
    return  model,history

path_img = "dataset3/"

files = os.listdir(path_img)

imagens = []
classes = []
for i in files:
    imagens = imagens + [imread(path_img + i)]
    classes = classes + [int(str(i.split("_")[0]))]

for i in range(len(imagens)):
    imagens[i] = resize(imagens[i], (imagens[i].shape[0] // 2, imagens[i].shape[1] // 2))

for i in range(len(imagens)):
   imagens[i] = imagens[i].flatten()

x_treino,x_teste,y_treino,y_teste = train_test_split(imagens,classes, test_size = 0.3, shuffle = True)
#y_treino = to_categorical(y_treino,num_classes = 2)
#y_teste = to_categorical(y_teste,num_classes = 2)
x_treino = np.array(x_treino)
y_treino = np.array(y_treino)
x_teste = np.array(x_teste)
y_teste = np.array(y_teste)

x_treino = x_treino.reshape([-1,50, 50,1])
x_teste = x_teste.reshape([-1,50, 50,1])



model,history = modelo(32,16,8)
#model2,history2 = modelo2(64,32,16,20)


scores = model.evaluate(x_treino, y_treino, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#scores2 = model2.evaluate(x_treino, y_treino, verbose=0)
#print("%s: %.2f%%" % (model2.metrics_names[1], scores2[1]*100))


model_json = model.to_json()
with open("modelo.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelo.h5")
print("Saved model to disk")

for key in history.history.keys():
    print(key)

pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test')
pyplot.legend()
pyplot.show()