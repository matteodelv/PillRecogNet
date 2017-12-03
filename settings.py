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