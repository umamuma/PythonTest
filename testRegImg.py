from __future__ import absolute_import, division, print_function

import tensorflow as tf 
from tensorflow import keras

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


img=mpimg.imread('001.JPG')
img = img[:,:,1]
plt.imshow(img, cmap="gray")
imShape = img.shape
#%%

inData = img.resapenp.zeros([imLen*imLen,2], np.float32)
outData = np.zeros([imLen*imLen], np.float32)

for xi in range(imLen):
	for yi in range(imLen):
		x = xi/imLen
		y  = yi/imLen
		inData[yi*imLen + xi,0] = x
		inData[yi*imLen + xi,1] = y
		if math.pow(x-0.5,2) + math.pow(y-0.5,2) < (0.4*0.4) :
			outData[yi*imLen + xi] = 1

mean = inData.mean(axis=0)
std = inData.std(axis=0)
inData = (inData - mean) / std

# Shuffle the training set
order = np.argsort(np.random.random(outData.shape))
inDataR = inData[order]
outDataR = outData[order]
		
def toImg(data,x,y):
	imData = data.reshape(x,y)
	return imData

def printImg(data):	
	plt.figure()
	plt.imshow(toImg(data,imLen, imLen))
	
printImg(outData)


def build_model():
  model = keras.Sequential([
    keras.layers.Dense(10, activation=tf.nn.relu,
                       input_shape=(inData.shape[1],)),
    #keras.layers.Dense(10, activation=tf.nn.relu),
	#keras.layers.Dense(10, activation=tf.nn.relu),
	keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)
  
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 50 == 0: print('')
    print('.', end='')

#%%
EPOCHS = 150

# Store training stats
history = model.fit(np.concatenate((inDataR,inData)), np.concatenate((outDataR,outData)), epochs=EPOCHS,
                    validation_split=0.5, verbose=0,
                    callbacks=[PrintDot()])

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error ')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, max(max(history.history['val_mean_absolute_error']), max(history.history['mean_absolute_error']))])

plot_history(history)

[loss, mae] = model.evaluate(inData, outData, verbose=0)

print("Testing set Mean Abs Error: {:7.2f}".format(mae * 1000))

outData_ = model.predict(inData).flatten()
printImg(outData_)
#%%
res = outData_-outData
res = res- min(res)
res = res/max(res)
printImg(res)