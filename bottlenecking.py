from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.applications import VGG16
import os
import numpy as np

# dimensions of our images.
img_width, img_height = 224, 224

train_bottlenecks_path = 'bottlenecks_train.npy'
validation_bottlenecks_path = 'bottlenecks_validation.npy'
top_model_weights_path = 'custom_layers_bottlenecks.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
plots_dir = 'plots'
nb_train_samples = 528
nb_validation_samples = 96
epochs = 100
batch_size = 16
num_classes = 12


def meanSubtraction(x):
    x = x.astype(np.float32)
    means = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
    x -= means
    return x

def save_bottlebeck_features():
    print("Bottleneck features calculation started...")

    datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=meanSubtraction)

    model = VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size, verbose=1)
    np.save(open(train_bottlenecks_path, 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples // batch_size, verbose=1)
    np.save(open(validation_bottlenecks_path, 'w'), bottleneck_features_validation)

    print("Bottlenecks saved...")


def train_top_model():
    print("Top model training started...")

    train_per_class = nb_train_samples // num_classes
    valid_per_class = nb_validation_samples // num_classes

    train_data = np.load(open(train_bottlenecks_path))
    train_labels = np.array([0] * train_per_class + [1] * train_per_class + [2] * train_per_class + [3] * train_per_class + [4] * train_per_class + [5] * train_per_class + [6] * train_per_class + [7] * train_per_class + [8] * train_per_class + [9] * train_per_class + [10] * train_per_class + [11] * train_per_class)

    validation_data = np.load(open(validation_bottlenecks_path))
    validation_labels = np.array([0] * valid_per_class + [1] * valid_per_class + [2] * valid_per_class + [3] * valid_per_class + [4] * valid_per_class + [5] * valid_per_class + [6] * valid_per_class + [7] * valid_per_class + [8] * valid_per_class + [9] * valid_per_class + [10] * valid_per_class + [11] * valid_per_class)

    train_labels = to_categorical(train_labels, num_classes=num_classes)
    validation_labels = to_categorical(validation_labels, num_classes=num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = SGD(lr=1e-3, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print("Fitting...")

    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(validation_data, validation_labels), verbose=1)

    model.save_weights(top_model_weights_path)
    model.save('bottleneck_model.h5')

    print("Model and weights saved...")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plot
    plot.plot(history.history['acc'])
    plot.plot(history.history['val_acc'])
    plot.title("Feature Extraction Accuracy")
    plot.ylabel('Accuracy')
    plot.xlabel('Epoch')
    plot.legend(['Training', 'Validation'], loc='lower right')
    plot.savefig(os.path.join(plots_dir, 'bottleneck_accuracy.png'))
    plot.close()

    plot.plot(history.history['loss'])
    plot.plot(history.history['val_loss'])
    plot.title("Feature Extraction Loss")
    plot.ylabel('Loss')
    plot.xlabel('Epoch')
    plot.legend(['Training', 'Validation'], loc='upper right')
    plot.savefig(os.path.join(plots_dir, 'bottleneck_loss.png'))
    plot.close()


save_bottlebeck_features()
train_top_model()
print("Done!")
