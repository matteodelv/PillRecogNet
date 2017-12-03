from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model
from keras.callbacks import LearningRateScheduler
import os
import math

import settings as s
import utils as u


# File paths
fine_tuning_acc_plot_path = os.path.join(s.plots_dir, 'fine_tuning_accuracy.png')
fine_tuning_loss_plot_path = os.path.join(s.plots_dir, 'fine_tuning_loss.png')
fine_tuning_alr_plot_path = os.path.join(s.plots_dir, 'fine_tuning_alr.png')


# Build VGG16 network
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(s.img_width,s.img_height,3))

# Obtain custom top layers configuration
top_model = u.obtainNewTopLayers(base_model.output_shape[1:], s.num_classes)

# Load top layers weights from exported features
print("Loading top weights...")
top_model.load_weights(s.top_model_weights_path)

# Attach custom top layers to VGG16 network
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# Set VGG16 layers to be non trainable so that only the last
# convolutional block will be fine tuned
for layer in model.layers[:15]:
    layer.trainable = False

# Compile the merged network; lr = 0.0 means that no learning rate
# has been specified since Adaptive Learning Rate will be used
optimizer = SGD(lr=0.0, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Apply image augmentation to training samples and elaborate them
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, width_shift_range=0.02, height_shift_range=0.02, rotation_range=20, horizontal_flip=True, preprocessing_function=u.meanSubtraction)
train_generator = train_datagen.flow_from_directory(s.train_data_dir, target_size=(s.img_height, s.img_width), batch_size=s.batch_size, class_mode='categorical')

# Load validation images and elaborate them
valid_datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=u.meanSubtraction)
validation_generator = valid_datagen.flow_from_directory(s.validation_data_dir, target_size=(s.img_height, s.img_width), batch_size=s.batch_size, class_mode='categorical')

print("Getting validation class indices...")
print(validation_generator.class_indices)

# Define Adaptive Learning Rate function to be used as a callback
lrs = [0.001]
def ALR(epoch):
    initialLR = 0.001
    drop = 0.5
    epochsDrop = 17
    newLR = initialLR * math.pow(drop, math.floor((1+epoch)/epochsDrop))
    lrs.append(newLR)
    print("\nCurrent LR = {:.7f}".format(newLR))
    return newLR

# Prepare the ALR callback
lrSched = LearningRateScheduler(ALR)

# Start fine tuning...
print("Fitting...")
history = model.fit_generator(train_generator, steps_per_epoch=s.nb_train_samples // s.batch_size, epochs=s.ftEpochs, validation_data=validation_generator, validation_steps=s.nb_validation_samples // s.batch_size, verbose=1, callbacks=[lrSched])

model.save_weights(s.fine_tuned_weights_path)
model.save(s.fine_tuned_model_path)

print("Model and weights saved...")
print("Trying to evaluate...")

# Evaluate fine tuned model
metrics = model.evaluate_generator(validation_generator, steps=s.nb_validation_samples // s.batch_size)
statsDict = dict(zip(model.metrics_names, metrics))
print(statsDict)

# Generate and save fine tuning plots
print("Saving plots...")
legend = ['Training', 'Validation']
accData = [history.history['acc'], history.history['val_acc']]
lossData = [history.history['loss'], history.history['val_loss']]
u.plotGraph(accData, "Fine Tuning Accuracy", "Epoch", "Accuracy", legend, fine_tuning_acc_plot_path)
u.plotGraph(lossData, "Fine Tuning Loss", "Epoch", "Loss", legend, fine_tuning_loss_plot_path)
u.plotGraph([lrs], "Learning Rates", "Epoch", "Learning Rate", None, fine_tuning_alr_plot_path)

print("Done!")
