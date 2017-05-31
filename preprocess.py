# -*- coding: utf-8 -*-
# # Planet: Understanding the Amazon deforestation from Space challenge
# Special thanks to the kernel contributors of this challenge (especially @anokas and @Kaggoo) who helped me find a starting point for this notebook.
# 
# The whole code including the `data_helper.py` and `keras_helper.py` files are available on github [here](https://github.com/EKami/planet-amazon-deforestation) and the notebook can be found on the same github [here](https://github.com/EKami/planet-amazon-deforestation/blob/master/notebooks/amazon_forest_notebook.ipynb)
# 
# **If you found this notebook useful some upvotes would be greatly appreciated! :) **
# Start by adding the helper files to the python path

# Import required modules
import os
import gc
import h5py
import pandas as pd

import data_helper

#%matplotlib inline
#%config InlineBackend.figure_format = "retina"

if not os.path.isdir("results"):
    os.mkdir("results")

## Read data
train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file = data_helper.get_jpeg_data_files_paths()
labels_df = pd.read_csv(train_csv_file)
print(labels_df.head())

# <codecell>
img_resize = (64, 64) # The resize size of each image

# <markdowncell>

# # Data preprocessing
# Preprocess the data in order to fit it into the Keras model.
# 
# Due to the hudge amount of memory the resulting matrices will take, the preprocessing will be splitted into several steps:
#     - Preprocess training data (images and labels) and train the neural net with it
#     - Delete the training data and call the gc to free up memory
#     - Preprocess the first testing set
#     - Predict the first testing set labels
#     - Delete the first testing set
#     - Preprocess the second testing set
#     - Predict the second testing set labels and append them to the first testing set
#     - Delete the second testing set

# <codecell>
# Preprocess training data and convert labels to indicator vectors

x_train, y_train, y_map = data_helper.preprocess_train_data(train_jpeg_dir, train_csv_file, img_resize)

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))
print(y_map)

f = h5py.File("results/train_rGg.h5", "w")
x_train_dset = f.create_dataset("x_train", x_train.shape, dtype="f")
x_train_dset[:] = x_train

y_train_dset = f.create_dataset("y_train", y_train.shape, dtype="i")
y_train_dset[:] = y_train

y_map_str = ["{}={}".format(k,v) for k, v in y_map.items()]
f.create_dataset("y_map", (len(y_map_str),), "S25", y_map_str)
f.flush()
f.close()

del x_train, y_train
gc.collect()

# # to read y_map
#with h5py.File("results/train.h5", "r") as f:
#    my_array = f["y_map"][()].tolist()
#    y_map = {int(key):value for key, value in [tuple(x.split("=")) for x in my_array]}

# <codecell>
# Preprocess testing data

x_test, x_test_filename = data_helper.preprocess_test_data(test_jpeg_dir, img_resize)

print("x_test shape: {}".format(x_test.shape))

f = h5py.File("results/test_rGg.h5", "w")
x_test_dset = f.create_dataset("x_test", x_test.shape, dtype="f")
x_test_dset[:] = x_test

f.create_dataset("x_test_filename", (len(x_test_filename),), "S20", x_test_filename)
f.flush()
f.close()

del x_test
gc.collect()

# # to read filename
#with h5py.File("results/test.h5", "r") as f:
#    my_array = f["x_test_filename"][()].tolist()

# <codecell>
# Preprocess additionnal testing data (updated on 05/05/2017 on Kaggle)

x_test, x_test_filename_additional = data_helper.preprocess_test_data(test_jpeg_additional, img_resize)

print("x_test additional shape: {}".format(x_test.shape))

f = h5py.File("results/test_additional_rGg.h5", "w")
x_test_dset = f.create_dataset("x_test", x_test.shape, dtype="f")
x_test_dset[:] = x_test

f.create_dataset("x_test_filename", (len(x_test_filename_additional),), "S20", x_test_filename_additional)
f.flush()
f.close()

del x_test
gc.collect()

# # to read filename
#with h5py.File("results/test_additional.h5", "r") as f:
#    my_array = f["x_test_filename"][()].tolist()

print("Preprocessing Done")