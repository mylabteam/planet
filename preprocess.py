# -*- coding: utf-8 -*-
# # Planet: Understanding the Amazon deforestation from Space challenge
# Special thanks to the kernel contributors of this challenge (especially @anokas and @Kaggoo) who helped me find a starting point for this notebook.
# 
# The whole code including the `data_helper.py` and `keras_helper.py` files are available on github [here](https://github.com/EKami/planet-amazon-deforestation) and the notebook can be found on the same github [here](https://github.com/EKami/planet-amazon-deforestation/blob/master/notebooks/amazon_forest_notebook.ipynb)
# 
# **If you found this notebook useful some upvotes would be greatly appreciated! :) **
# Start by adding the helper files to the python path

import os
import h5py
import pandas as pd

import data_helper


if not os.path.isdir("results"):
    os.mkdir("results")

## Read data
_, _, _, train_csv_file, train_tif_dir, test_tif_dir = data_helper.get_data_files_paths()
labels_df = pd.read_csv(train_csv_file)
print(labels_df.head())

# <codecell>
img_resize = (64, 64) # The resize size of each image
h5_train_file = "results/train_tif.h5"
h5_test_file = "results/test_tif.h5"
h5_test_add_file = "results/test_additional_tif.h5"
# <markdowncell>

# # Data preprocessing
# Preprocess the data in order to fit it into the Keras model.

# <codecell>
# Preprocess training data and convert labels to indicator vectors
files_path, tags_list, labels_map = data_helper.get_file_tag_list(train_csv_file, train_tif_dir)
data_helper.preprocess_train_data(files_path, tags_list, labels_map, h5_train_file, 
                                  data_helper._train_transform_to_matrices, img_resize, 4)
with h5py.File(h5_train_file, "r") as f:
    my_array = f["y_map"][()].tolist()
    y_map = {int(key):value for key, value in [tuple(x.split("=")) for x in my_array]}
    print("x_train shape: {}".format(f["x_train"].shape))
    print("y_train shape: {}".format(f["y_train"].shape))
    print(y_map)

# <codecell>
# Preprocess testing data and get list of filenames
data_helper.preprocess_test_data(test_tif_dir, h5_test_file, 
                                 data_helper._test_transform_to_matrices, img_resize, 4)
with h5py.File(h5_test_file, "r") as f:
    x_test_filename = f["x_test_filename"][()].tolist() 
    print("x_test shape         : {}".format(f["x_test"].shape))
    print("x_test_filename shape: {}".format(len(x_test_filename)))

print("Preprocessing Done")