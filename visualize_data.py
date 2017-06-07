# -*- coding: utf-8 -*-
# # Planet: Understanding the Amazon deforestation from Space challenge
# Special thanks to the kernel contributors of this challenge (especially @anokas and @Kaggoo) who helped me find a starting point for this notebook.
# 
# The whole code including the `data_helper.py` and `keras_helper.py` files are available on github [here](https://github.com/EKami/planet-amazon-deforestation) and the notebook can be found on the same github [here](https://github.com/EKami/planet-amazon-deforestation/blob/master/notebooks/amazon_forest_notebook.ipynb)
# 
# **If you found this notebook useful some upvotes would be greatly appreciated! :) **
# Start by adding the helper files to the python path

# Import required modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
from keras.utils.io_utils import HDF5Matrix

from plotly.offline import plot
import plotly.graph_objs as go

import data_helper

# <codecell>
## Inspect image labels
# Visualize what the training set looks like
train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file, _, _ = data_helper.get_data_files_paths()
labels_df = pd.read_csv(train_csv_file)
print(labels_df.head())

# <codecell>
# Each image can be tagged with multiple tags, lets list all uniques tags

# Print all unique tags
from itertools import chain
labels_list = list(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values]))
labels_set = set(labels_list)
print("There is {} unique labels including {}".format(len(labels_set), labels_set))


# ### Repartition of each labels
# Histogram of label instances
labels_s = pd.Series(labels_list).value_counts() # To sort them by count
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=labels_s, y=labels_s.index, orient='h')


# ## Images
# Visualize some chip images to know what we are dealing with.
# Lets vizualise 1 chip for the 17 images to get a sense of their differences.

# <codecell>

images_title = [labels_df[labels_df['tags'].str.contains(label)].iloc[i]['image_name'] + '.jpg' 
                for i, label in enumerate(labels_set)]

plt.rc('axes', grid=False)
_, axs = plt.subplots(5, 4, sharex='col', sharey='row', figsize=(15, 20))
axs = axs.ravel()

for i, (image_name, label) in enumerate(zip(images_title, labels_set)):
    img = mpimg.imread(train_jpeg_dir + '/' + image_name)
    axs[i].imshow(img)
    axs[i].set_title('{} - {}'.format(image_name, label))

# <codecell>
# Co-occurance
h5_train_file = "results/train_jpg_rgb.h5"
h5_test_file = "results/test_jpg_rgb.h5"
with h5py.File(h5_train_file, "r") as f:
    my_array = f["y_map"][()].tolist()
    y_map = {int(key):value for key, value in [tuple(x.split("=")) for x in my_array]}
y_train = HDF5Matrix(h5_train_file, "y_train")
Y = np.array(y_train)
# <codecell>
cooc = np.zeros((17,17))
order = np.argsort(Y.sum(axis=0))
tags = np.array(y_map.values())
tags = tags[order]
for i in range(17):
    for j in range(17):
        cooc[i,j] = np.logical_and(Y[:,i], Y[:,j]).sum()
cooc=cooc[order][:,order]

idx = np.array([0,1,2,3,4,5,6,8,9,10,11,12,13])
[t for t in tags[idx]]

trace = go.Heatmap(z=cooc[idx][:,idx],x=tags[idx],y=tags[idx])
plot([trace], filename='cooccurance-heatmap.html')
