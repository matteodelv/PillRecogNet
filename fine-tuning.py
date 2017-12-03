from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import LearningRateScheduler
import numpy as np
import os
import math

top_model_weights_path = 'custom_layers_bottlenecks.h5'
img_width, img_height = 224, 224

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
plots_dir = 'plots'
nb_train_samples = 528
nb_validation_samples = 96
epochs = 50
batch_size = 16
num_classes = 12

# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width,img_height,3))
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
print("Loading top weights...")
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
# model.add(top_model)
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# set all VGG16 layers as non trainable except from the last conv block
# Doing so, only layers from block5_conv3 (included) to the fully connected layers will be fine tuned
for layer in model.layers[:15]: # Usare :11 per allenare anche il penultimo blocco convoluzionale
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.0, momentum=0.9),
              metrics=['accuracy'])

def meanSubtraction(x):
    x = x.astype(np.float32)
    means = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
    x -= means
    return x

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.02,
    height_shift_range=0.02,
    rotation_range=20,
    horizontal_flip=True, preprocessing_function=meanSubtraction)

test_datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=meanSubtraction)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

print("Getting validation class indices...")
print(validation_generator.class_indices)

#model.summary()

#lrReducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)
#stopper = EarlyStopping(monitor='val_acc', patience=10, verbose=1)

lrs = [0.001]
def ALR(epoch):
    initialLR = 0.001
    drop = 0.5
    epochsDrop = 17
    newLR = initialLR * math.pow(drop, math.floor((1+epoch)/epochsDrop))
    lrs.append(newLR)
    print("\nCurrent LR = {:.7f}".format(newLR))
    return newLR

lrSched = LearningRateScheduler(ALR)

print("Fitting...")
# fine-tune the model
history = model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs, validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size, verbose=1, callbacks=[lrSched])

model.save_weights('fine-tuned-weights.h5')
model.save('fine-tuned-model.h5')

print("Model and weights saved...")
print("Trying to evaluate...")

metrics = model.evaluate_generator(validation_generator, steps=nb_validation_samples // batch_size)
#print(metrics)
#print(model.metrics_names)

statsDict = dict(zip(model.metrics_names, metrics))
print(statsDict)

print(lrs)

print("Saving plots...")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
plot.plot(history.history['acc'])
plot.plot(history.history['val_acc'])
plot.title("Fine Tuning Accuracy")
plot.ylabel('Accuracy')
plot.xlabel('Epoch')
plot.legend(['Training', 'Validation'], loc='lower right')
plot.savefig(os.path.join(plots_dir, 'fine_tuning_accuracy.png'))
plot.close()

plot.plot(history.history['loss'])
plot.plot(history.history['val_loss'])
plot.title("Fine Tuning Loss")
plot.ylabel('Loss')
plot.xlabel('Epoch')
plot.legend(['Training', 'Validation'], loc='upper right')
plot.savefig(os.path.join(plots_dir, 'fine_tuning_loss.png'))
plot.close()

plot.plot(lrs)
plot.title("Learning Rates")
plot.ylabel("Learning Rate")
plot.xlabel("Epoch")
plot.savefig(os.path.join(plots_dir, 'fine-tuning-alr.png'))
plot.close()

print("Done!")
