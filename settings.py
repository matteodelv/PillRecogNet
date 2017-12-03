import os

# Shared training settings

# Image size accepted by CNN
img_width, img_height = 224, 224

# Training settings
# Train and validation sample numbers MUST be divisible by batch_size
nb_train_samples = 528
nb_validation_samples = 96
botEpochs = 100
ftEpochs = 50
batch_size = 16
num_classes = 12

# folder paths
plots_dir = 'plots'
results_dir = 'results'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

top_model_weights_path = os.path.join(results_dir, 'custom_layers_bottlenecks.h5')
top_model_model_path = os.path.join(results_dir, 'bottleneck_model.h5')

fine_tuned_model_path = os.path.join(results_dir, "fine-tuned-model.h5")
fine_tuned_weights_path = os.path.join(results_dir, "fine-tuned-weights.h5")