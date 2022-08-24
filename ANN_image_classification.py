import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
in keras mnist dataset train and test data are already divided
so directly load the dataset from keras dataset and put in inside 
x_train, x_test, y_train and y_test
'''
import keras.datasets.mnist as mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

'''
1. Data preprocessing
'''
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(np.max(x_train), np.min(x_train), np.max(y_train), np.min(y_train))

def image(x, y, index):
    plt.figure(figsize=(5,5))
    plt.imshow(x[index], cmap='gray')
    plt.xlabel(y[index])
    plt.show()

image(x_train, y_train, 0)
'''
each training image has pixel value between 0-255
height X width of 28X28 with black and white since no layer is present
number of output is 10 (0-9)
60000 train set and 10000 is test set
'''
'''
now, normalize the pixel value in between 0-1, instead 0-255 for the faster training experience
'''
x_train, x_test = x_train/255.0, x_test/255.0
print(x_train.shape, x_test.shape)

'''
2. model building

I need the entire operation will happen in a specific sequence from layer-1 to
layer-2 to layer-3. therefore i have used sequence.

First i need to flatten the image into 1D-Data because in deep learning model
its take data in 1D-format

Since, every node of 1st-layer is densely connected with another 2nd-layer.
therefore using Dense with no of node=128 and activation function as ReLU

formation of 2nd-layer will be based upon dropout. Those node which doesn't
make significant improvement will be removed by the rate of 0.2

formation of 3rd layer will be densely connected with 2nd-layer therefore
using Dense with no. of node = 10, because at the output i'm expecting 10 output
(depending upon the problem, how many output you needed) and softmax function as
activation. Most of time in image classification uses softmax function.
'''
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='Softmax')
])

model.summary()

'''
3. compiling the Model

Before training the model in ANN, we need to compile the model, in compiling will going to use
adam as our optimizer which will autoupdate the weights of the model and minimize the loss at
the end of the output.

Addition to that we will be using "Sparse categorical crossentropy" as a loss function which will
act as guide to optimizer (adam). to understand the accuracy of the model will be using 
matrices = "sparse categorical accuracy or accuracy depending upon multiple output or binary output
respectively. 
'''

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')

'''
4. Training of Model and evaluation of model
'''
training = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
evaluating = model.evaluate(x_test, y_test)

'''
Graphs

for plotting any graphs we need to put training model in some variable i.e. training

epochs means, how amy time are you training your model
'''

plt.plot(training.history['loss'], label = 'loss')
plt.plot(training.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()

plt.plot(training.history['accuracy'], label = 'accuracy')
plt.plot(training.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()

'''
5. prediction

after training and evaluation the model will be predicted on x-test data.
since i have converted the pixel in between 0-1 and there is 10 output node therefore getting output in 
list format of 10 outputs. 

in order to get the correct output i have used np.argmax method to show which index position will be having 
high value(highest probability) among the output list.
'''
y_pred = model.predict(x_test)
print(y_pred)

y_pred_convert = []
for i in y_pred:
    y_pred_convert.append(np.argmax(i))
print(y_pred_convert)

'''
6. Output dataframe
'''
y_test = y_test.reshape(-1, )
y_pred_convert = np.array(y_pred_convert)
y_pred_convert = y_pred_convert.reshape(-1, )

print(len(y_pred), len(y_test))

output_df = pd.DataFrame(
    {
        'actual':y_test, 'prediction':y_pred_convert
    }
)

print(output_df)

'''
7. Confusion matrix
'''
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_convert)
print(cm)

a_s = accuracy_score(y_test, y_pred_convert)
print(a_s)
