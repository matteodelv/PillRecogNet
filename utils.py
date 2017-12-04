import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
import os
import errno


# Preprocessing function to subtract ImageNet mean RGB values from new images
# to keep weights coherent during transfer learning
def meanSubtraction(x):
	x = x.astype(np.float32)
	means = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
	x -= means
	return x

# Helper function to get new top layers configuration
def obtainNewTopLayers(input_shape, num_classes):
	model = Sequential()
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	return model

# Helper function to plot and save graphs about the training
def plotGraph(graphData, title, xlabel, ylabel, legend, savePath):
	for (i, data) in enumerate(graphData):
		plot.plot(data)
	plot.title(title)
	plot.ylabel(ylabel)
	plot.xlabel(xlabel)
	if legend != None:
		plot.legend(legend, loc='lower right')
	plot.savefig(savePath)
	plot.close()

# Helper function to organize folder structure
def createDirIfNotExisting(path):
	try:
		os.makedirs(path)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

# Raise exception if training data is not found
def checkDirs(paths):
	for (i, path) in enumerate(paths):
		if not os.path.isdir(path):
			raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), path)