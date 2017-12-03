from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.applications import VGG16
import os
import numpy as np

from utils import meanSubtraction, plotGraph, obtainNewTopLayers, createDirIfNotExisting
import settings as s


# file paths
bottleneck_train_datapath = os.path.join(s.results_dir, 'bottlenecks_train.npy')
bottleneck_valid_datapath = os.path.join(s.results_dir, 'bottlenecks_validation.npy')
bottleneck_accuracy_plot_path = os.path.join(s.plots_dir, 'bottleneck_accuracy.png')
bottleneck_loss_plot_path = os.path.join(s.plots_dir, 'bottleneck_loss.png')


def save_bottlebeck_features():
	print("Bottleneck features calculation started...")

	# Load VGG16 network without top layers to act as a feature extractor
	model = VGG16(include_top=False, weights='imagenet')

	# Get features for train and validation images and save them
	datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=meanSubtraction)
	generator = datagen.flow_from_directory(s.train_data_dir, target_size=(s.img_width, s.img_height), batch_size=s.batch_size, class_mode=None, shuffle=False)
	bottleneck_features_train = model.predict_generator(generator, s.nb_train_samples // s.batch_size, verbose=1)
	np.save(open(bottleneck_train_datapath, 'w'), bottleneck_features_train)

	generator = datagen.flow_from_directory(s.validation_data_dir, target_size=(s.img_width, s.img_height), batch_size=s.batch_size, class_mode=None, shuffle=False)
	bottleneck_features_validation = model.predict_generator(generator, s.nb_validation_samples // s.batch_size, verbose=1)
	np.save(open(bottleneck_valid_datapath, 'w'), bottleneck_features_validation)

	print("Bottlenecks saved...")


def train_top_model():
	print("Top model training started...")

	train_per_class = s.nb_train_samples // s.num_classes
	valid_per_class = s.nb_validation_samples // s.num_classes

	# Load saved features from bottlenecks
	train_data = np.load(open(bottleneck_train_datapath))
	train_labels = np.array([0] * train_per_class + [1] * train_per_class + [2] * train_per_class + [3] * train_per_class + [4] * train_per_class + [5] * train_per_class + [6] * train_per_class + [7] * train_per_class + [8] * train_per_class + [9] * train_per_class + [10] * train_per_class + [11] * train_per_class)

	validation_data = np.load(open(bottleneck_valid_datapath))
	validation_labels = np.array([0] * valid_per_class + [1] * valid_per_class + [2] * valid_per_class + [3] * valid_per_class + [4] * valid_per_class + [5] * valid_per_class + [6] * valid_per_class + [7] * valid_per_class + [8] * valid_per_class + [9] * valid_per_class + [10] * valid_per_class + [11] * valid_per_class)

	train_labels = to_categorical(train_labels, num_classes=s.num_classes)
	validation_labels = to_categorical(validation_labels, num_classes=s.num_classes)

	# Create new top layers
	model = obtainNewTopLayers(train_data.shape[1:], s.num_classes)

	# Compile the model using Stocastic Gradient Descent and a low learning rate
	optimizer = SGD(lr=1e-3, momentum=0.9)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	# Start training...
	print("Fitting...")
	history = model.fit(train_data, train_labels, epochs=s.botEpochs, batch_size=s.batch_size, validation_data=(validation_data, validation_labels), verbose=1)

	model.save_weights(s.top_model_weights_path)
	model.save(s.top_model_model_path)

	print("Model and weights saved...")

	# Create graphs
	legend = ['Training', 'Validation']
	accData = [history.history['acc'], history.history['val_acc']]
	lossData = [history.history['loss'], history.history['val_loss']]
	plotGraph(accData, "Feature Extraction Accuracy", "Epoch", "Accuracy", legend, bottleneck_accuracy_plot_path)
	plotGraph(lossData, "Feature Extraction Loss", "Epoch", "Loss", legend, bottleneck_loss_plot_path)



if __name__ == '__main__':
	createDirIfNotExisting(s.plots_dir)
	createDirIfNotExisting(s.results_dir)
	#save_bottlebeck_features()
	train_top_model()
	print("Done!")
