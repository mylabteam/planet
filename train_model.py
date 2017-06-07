# -*- coding: utf-8 -*-
# # Planet: Understanding the Amazon deforestation from Space challenge
# Special thanks to the kernel contributors of this challenge (especially @anokas and @Kaggoo) who helped me find a starting point for this notebook.
# 
# The whole code including the `data_helper.py` and `keras_helper.py` files are available on github [here](https://github.com/EKami/planet-amazon-deforestation) and the notebook can be found on the same github [here](https://github.com/EKami/planet-amazon-deforestation/blob/master/notebooks/amazon_forest_notebook.ipynb)
# 
# **If you found this notebook useful some upvotes would be greatly appreciated! :) **
# Start by adding the helper files to the python path

# Import required modules
import h5py
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import fbeta_score
from keras.utils.io_utils import HDF5Matrix
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain

import data_helper
from keras_helper import AmazonKerasClassifier

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

# Print tensorflow version for reuse (the Keras module is used directly from the tensorflow framework)
print(tf.__version__)

## Inspect image labels
# Visualize what the training set looks like

_, _, _, train_csv_file, _, _ = data_helper.get_data_files_paths()

# <codecell>
h5_train_file = "results/train_jpg_rgb.h5"
h5_test_file = "results/test_jpg_rgb.h5"
#h5_test_add_file = "results/test_additional_rGg.h5"
model_filepath="weights.best_jpg_rgb"
submission_file="submission_file_jpg_rgb.csv"

# Hyperparameters: choose your hyperparameters below for training. 
img_resize = (64, 64) # The resize size of each image
img_channels = 3
validation_split_size = 0.2
batch_size = 128

# <codecell>
# <markdowncell>
# ## Create a checkpoint
# Creating a checkpoint saves the best model weights across all epochs in the training process. This ensures that we will always use only the best weights when making our predictions on the test set rather than using the default which takes the final score from the last epoch. 
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(model_filepath+".hdf5", monitor='val_acc', verbose=1, save_best_only=True)

# <codecell>
# Load data
# Read y_map
with h5py.File(h5_train_file, "r") as f:
    N_train = f["x_train"].shape[0]
    my_array = f["y_map"][()].tolist()
    y_map = {int(key):value for key, value in [tuple(x.split("=")) for x in my_array]}

N_split = int(round(N_train * (1-validation_split_size)))
#N_split = 32256
#N_train = 40448
x_train = HDF5Matrix(h5_train_file, "x_train", start=0, end=N_split)
y_train = HDF5Matrix(h5_train_file, "y_train", start=0, end=N_split)

x_valid = HDF5Matrix(h5_train_file, "x_train", start=N_split, end=N_train)
y_valid = HDF5Matrix(h5_train_file, "y_train", start=N_split, end=N_train)


# <codecell>
# <markdowncell>
# ## Define and Train model
# Here we define the model and begin training. 
# Note that we have created a learning rate annealing schedule with a series of learning rates as defined in the array `learn_rates` and corresponding number of epochs for each `epochs_arr`. Feel free to change these values if you like or just use the defaults. 
classifier = AmazonKerasClassifier()
#classifier.load_model(model_filepath+".json")       # load model
classifier.add_conv_layer(img_resize, img_channels)
classifier.add_flatten_layer()
classifier.add_ann_layer(len(y_map))
classifier.save_model(model_filepath+".json")

train_losses, val_losses = [], []
epochs_arr = [10, 5, 5]
learn_rates = [0.001, 0.0001, 0.00001]
for learn_rate, epochs in zip(learn_rates, epochs_arr):
    tmp_train_losses, tmp_val_losses, fbeta_sc = classifier.train_model(x_train, y_train, x_valid, y_valid,
                                                                           learn_rate, epochs, batch_size, 
                                                                           train_callbacks=[checkpoint])
    train_losses += tmp_train_losses
    val_losses += tmp_val_losses
    y_pred = classifier.predict(x_valid)
    f2samples = fbeta_score(np.array(y_valid), y_pred > 0.2, beta=2, average='samples')
    print("F2 samples = {}".format(f2samples))

# <codecell>
# ## Load Best Weights
# Here you should load back in the best weights that were automatically saved by ModelCheckpoint during training
classifier.load_weights(model_filepath+".hdf5")
print("Weights loaded")
y_pred = classifier.predict(x_valid)
f2 = fbeta_score(np.array(y_valid), y_pred > 0.2, beta=2, average=None)

sns.barplot(x=f2,y=y_map.values())

f2samples = fbeta_score(np.array(y_valid), y_pred > 0.2, beta=2, average="samples")
print("F2 samples = {}".format(f2samples))
# <codecell>
# ## Monitor the results
# Check that we do not overfit by plotting the losses of the train and validation sets
plt.figure()
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend();

# <codecell>
# Look at our fbeta_score
print("mean Fbeta score = {}".format(fbeta_sc))
gc.collect()

# <codecell>
# Predict the labels of our x_test images
x_test = HDF5Matrix(h5_test_file, "x_test")
# Read filename
with h5py.File(h5_test_file, "r") as f:
    x_test_filename = f["x_test_filename"][()].tolist()
# Predict the labels of our x_test images
predictions = classifier.predict(x_test)
gc.collect()

# <codecell>
#predictions = np.vstack((predictions, new_predictions))
#x_test_filename = np.hstack((x_test_filename, x_test_filename_additional))
print("Predictions shape: {}\nFiles name shape: {}\n1st predictions entry:\n{}".format(predictions.shape, 
                                                                              np.array(x_test_filename).shape,
                                                                              predictions[0]))

# <codecell>
# TODO complete
tags_pred = np.array(predictions).T
_, axs = plt.subplots(5, 4, figsize=(15, 20))
axs = axs.ravel()

for i, tag_vals in enumerate(tags_pred):
    sns.boxplot(tag_vals, orient='v', palette='Set2', ax=axs[i]).set_title(y_map[i])

# <markdowncell>
# <codecell>
# For now we'll just put all thresholds to 0.2 
thresholds = np.repeat(0.2, len(y_map.values()))
# Now lets map our predictions to their tags and use the thresholds we just retrieved
predicted_labels = []
for i in range(predictions.shape[0]):
    predicted_labels.append( [y_map[x] for x in np.where(predictions[i] > thresholds)[0]] )

# <codecell>
# Finally lets assemble and visualize our prediction for the test dataset
tags_list = [' '.join(tags) for i, tags in enumerate(predicted_labels)]
final_data = [[filename.split(".")[0], tags] for filename, tags in zip(x_test_filename, tags_list)]
final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
print(final_df.head())
# And save it to a submission file
final_df.to_csv(submission_file, index=False)
classifier.close()

# <codecell>
tags_s = pd.Series(list(chain.from_iterable(predicted_labels))).value_counts()
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=tags_s, y=tags_s.index, orient='h');
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_width()), (p.get_width()+50, p.get_y()+0.5))
# <markdowncell>
# If there is a lot of `primary` and `clear` tags, this final dataset may be legit...
# <markdowncell>
