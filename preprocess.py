# -*- coding: utf-8 -*-
# # Planet: Understanding the Amazon deforestation from Space challenge
# Special thanks to the kernel contributors of this challenge (especially @anokas and @Kaggoo) who helped me find a starting point for this notebook.
# 
# The whole code including the `data_helper.py` and `keras_helper.py` files are 
# available on github [here](https://github.com/EKami/planet-amazon-deforestation) and 
# the notebook can be found on the same 
# github [here](https://github.com/EKami/planet-amazon-deforestation/blob/master/notebooks/amazon_forest_notebook.ipynb)
# 
# **If you found this notebook useful some upvotes would be greatly appreciated! :) **
# Start by adding the helper files to the python path

import os
import h5py
import pandas as pd
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import data_helper
import random
random.seed(42)


if not os.path.isdir("results"):
    os.mkdir("results")

## Read data
train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file, train_tif_dir, test_tif_dir = data_helper.get_data_files_paths()
labels_df = pd.read_csv(train_csv_file)
print(labels_df.head())

# <codecell>
img_resize = (64, 64) # The resize size of each image
img_channels = 3
h5_train_file = "results/train_jpg_rgb.h5"
h5_test_file = "results/test_jpg_rgb.h5"
#h5_test_add_file = "results/test_additional_tif.h5"
# <markdowncell>

# # Data preprocessing
# Preprocess the data in order to fit it into the Keras model.

# <codecell>
# Preprocess training data and convert labels to indicator vectors
jpg_path_train, tags_list, labels_map = data_helper.get_file_tag_list(train_csv_file, train_jpeg_dir, ext='jpg')

# Preprocessing training data
f = h5py.File(h5_train_file, "w")
N = len(jpg_path_train)
x_train_dset = f.create_dataset("x_train", (N, img_resize[0], img_resize[1], img_channels), dtype="f")
y_train_dset = f.create_dataset("y_train", (N, len(labels_map)), dtype="i")

x_train = []
y_train = []
checkpnt = range(0,N,1000)
if checkpnt[-1] != N:
    checkpnt.append(N)
# Multiprocess transformation, the map() function take a function as a 1st argument
# and the argument to pass to it as the 2nd argument. These arguments are processed
# asynchronously on threads defined by process_count and their results are stored in
# the x_train and y_train lists
for i in range(1,len(checkpnt)):
    print("Preprocess Training data {}/{}".format(checkpnt[i-1], N))
    filespath = jpg_path_train[checkpnt[i-1]:checkpnt[i]]
    tagslist = tags_list[checkpnt[i-1]:checkpnt[i]]
    with ThreadPoolExecutor(cpu_count()) as pool:
        for img_array, targets in tqdm(pool.map(data_helper._train_transform_to_matrices,
                                                [(file_path, tag, labels_map, img_resize, None)
                                                 for file_path, tag in zip(filespath, tagslist)]),
                                       total=checkpnt[i]-checkpnt[i-1], mininterval=1.0):
            x_train.append(img_array)
            y_train.append(targets)
        x_train_dset[checkpnt[i-1]:checkpnt[i],:,:,:] = x_train
        y_train_dset[checkpnt[i-1]:checkpnt[i],:] = y_train
        x_train = []
        y_train = []
# save y_map
y_map = {v: k for k, v in labels_map.items()}
y_map_str = ["{}={}".format(k,v) for k, v in y_map.items()]
f.create_dataset("y_map", (len(y_map_str),), "S100", y_map_str)
f.flush()
f.close()
print("Done")

# check training preprocessing results
with h5py.File(h5_train_file, "r") as f:
    my_array = f["y_map"][()].tolist()
    y_map = {int(key):value for key, value in [tuple(x.split("=")) for x in my_array]}
    print("x_train shape: {}".format(f["x_train"].shape))
    print("y_train shape: {}".format(f["y_train"].shape))
    print(y_map)

# <codecell>
# Preprocess testing data and get list of filenames
print("Get test JPEG mean std")
jpg_path_test = ["{}/{}".format(test_jpeg_dir, filename) for filename in os.listdir(test_jpeg_dir)] + \
                ["{}/{}".format(test_jpeg_additional, filename) for filename in os.listdir(test_jpeg_additional)]

# Preprocessing testing data
f = h5py.File(h5_test_file, "w")
files_name = jpg_path_test
N = len(files_name)
x_test_dset = f.create_dataset("x_test", (N, img_resize[0], img_resize[1], img_channels), dtype="f")
x_test_filename_dset = f.create_dataset("x_test_filename", (N,), "S100")

x_test = []
x_test_filename = []
checkpnt = range(0,N,1000)
if checkpnt[-1] != N:
    checkpnt.append(N)
# Multiprocess transformation, the map() function take a function as a 1st argument
# and the argument to pass to it as the 2nd argument. These arguments are processed
# asynchronously on threads defined by process_count and their results are stored in
# the x_test and x_test_filename lists
for i in range(1,len(checkpnt)):
    print("Preprocess Testing data {}/{}".format(checkpnt[i-1], N))
    filesname = files_name[checkpnt[i-1]:checkpnt[i]]
    with ThreadPoolExecutor(cpu_count()) as pool:
        for img_array, file_name in tqdm(pool.map(data_helper._test_transform_to_matrices,
                                                  [("", file_name, img_resize, None)
                                                   for file_name in filesname]),
                                         total=checkpnt[i]-checkpnt[i-1], mininterval=1.0):
            x_test.append(img_array)
            x_test_filename.append(file_name.split("/")[-1])
        x_test_dset[checkpnt[i-1]:checkpnt[i],:,:,:] = x_test
        x_test_filename_dset[checkpnt[i-1]:checkpnt[i]] = x_test_filename
        x_test = []
        x_test_filename = []
f.flush()
f.close()
print("Done.")

# check testing preprocessing results
with h5py.File(h5_test_file, "r") as f:
    x_test_filename = f["x_test_filename"][()].tolist() 
    print("x_test shape         : {}".format(f["x_test"].shape))
    print("x_test_filename shape: {}".format(len(x_test_filename)))

# <codecell>
print("Preprocessing Done")