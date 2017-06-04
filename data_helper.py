import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io
from skimage.transform import resize
from itertools import chain
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

def get_data_files_paths(data_root_folder=os.path.abspath("./")):
    """
    Returns the input file folders path
    
    :return: list of strings
        The input file paths as list [train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file]
    """
    train_jpeg_dir = os.path.join(data_root_folder, 'train-jpg')
    test_jpeg_dir = os.path.join(data_root_folder, 'test-jpg')
    test_jpeg_additional = os.path.join(data_root_folder, 'test-jpg-additional')
    train_csv_file = os.path.join(data_root_folder, 'train_v2.csv')
    train_tif_dir = os.path.join(data_root_folder, 'train-tif-v2')
    test_tif_dir = os.path.join(data_root_folder, 'test-tif-v2')
    return [train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file, train_tif_dir, test_tif_dir]


def get_file_tag_list(train_csv_file, train_set_folder, ext='tif'):
    labels_df = pd.read_csv(train_csv_file)
    labels = sorted(set(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values])))
    labels_map = {l: i for i, l in enumerate(labels)}

    files_path = []
    tags_list = []
    for file_name, tags in labels_df.values:
        files_path.append('{}/{}.{}'.format(train_set_folder, file_name, ext))
        tags_list.append(tags)
    return [files_path, tags_list, labels_map]


def _get_image(img, img_resize):
    img_array = resize(img, img_resize)
    #img_array = np.asarray(img.convert("RGB"), dtype=np.float32) / 255
    # find rGg chromaticity
    rgbsum = img_array.sum(axis=2) + np.finfo(float).eps
    r = img_array[:,:,0] / rgbsum;
    g = img_array[:,:,1] / rgbsum;
    b = img_array[:,:,2] / rgbsum;
    #rGg = np.concatenate((r[:,:,np.newaxis],g[:,:,np.newaxis],b[:,:,np.newaxis]), axis=2)
    
    img_array[:,:,0] = r;
    img_array[:,:,2] = g;
    return img_array


def _get_tif_image(img, img_resize):
    img = resize(img, img_resize)
    return img


def _train_transform_to_matrices(*args):
    """
    
    :param args: list of arguments
        file_path: string
            The path of the image
        tags: list of strings
            The associated tags
        labels_map: dict {int: string}
            The map between the image label and their id 
        img_resize: tuple (int, int)
            The resize size of the original image given by the file_path argument
    :return: img_array, targets
        img_array: Numpy array
            The image from the file_path as a numpy array resized with img_resize
        targets: Numpy array
            A 17 length vector
    """
    # Unpack the *args
    file_path, tags, labels_map, img_resize = list(args[0])
    img = io.imread(file_path)
    # Augment the image `img` here
    if file_path[-3:] == 'tif':
        img_array = _get_tif_image(img, img_resize)
    else:
        img_array = _get_image(img, img_resize)
    
    targets = np.zeros(len(labels_map))
    for t in tags.split(' '):
        targets[labels_map[t]] = 1
    return img_array, targets


def _test_transform_to_matrices(*args):
    """
    :param args: list of arguments
        test_set_folder: string
            The path of the all the test images
        file_name: string
            The name of the test image
        img_resize: tuple (int, int)
            The resize size of the original image given by the file_path argument
        :return: img_array, file_name
            img_array: Numpy array
                The image from the file_path as a numpy array resized with img_resize
            file_name: string
                The name of the test image
        """
    test_set_folder, file_name, img_resize = list(args[0])
    file_path = "{}/{}".format(test_set_folder, file_name)
    img = io.imread(file_path)
    # Augment the image `img` here
    if file_path[-3:] == 'tif':
        img_array = _get_tif_image(img, img_resize)
    else:
        img_array = _get_image(img, img_resize)
    return img_array, file_name


def preprocess_train_data(files_path, tags_list, labels_map, h5save, preprocfun, img_resize=(32, 32), img_channels=3, process_count=cpu_count()):
    """
    Transform the train images to ready to use data for the CNN 
    :param train_set_folder: the folder containing the images for training
    :param train_csv_file: the file containing the labels of the training images
    :param img_resize: the standard size you want to have on images when transformed to matrices
    :param process_count: the number of process you want to use to preprocess the data.
        If you run into issues, lower this number. Its default value is equal to the number of core of your CPU
    :return: The images matrices and labels as [x_train, y_train, labels_map]
        x_train: The X train values as a numpy array
        y_train: The Y train values as a numpy array
        labels_map: The mapping between the tags labels and their indices
    """
    f = h5py.File(h5save, "w")
    N = len(files_path)
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
        filespath = files_path[checkpnt[i-1]:checkpnt[i]]
        tagslist = tags_list[checkpnt[i-1]:checkpnt[i]]
        with ThreadPoolExecutor(process_count) as pool:
            for img_array, targets in tqdm(pool.map(preprocfun,
                                                    [(file_path, tag, labels_map, img_resize)
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
    f.create_dataset("y_map", (len(y_map_str),), "S25", y_map_str)
    f.flush()
    f.close()
    print("Done")
    return


def preprocess_test_data(test_set_folder, h5save, preprocfun, img_resize=(32, 32), img_channels=3, process_count=cpu_count()):
    """
    Transform the images to ready to use data for the CNN
    :param test_set_folder: the folder containing the images for testing
    :param img_resize: the standard size you want to have on images when transformed to matrices
    :param process_count: the number of process you want to use to preprocess the data.
        If you run into issues, lower this number. Its default value is equal to the number of core of your CPU
    :return: The images matrices and labels as [x_test, x_test_filename]
        x_test: The X test values as a numpy array
        x_test_filename: The files name of each test images in the same order as the x_test arrays
    """
    f = h5py.File(h5save, "w")
    files_name = os.listdir(test_set_folder)
    N = len(files_name)
    x_test_dset = f.create_dataset("x_test", (N, img_resize[0], img_resize[1], img_channels), dtype="f")
    x_test_filename_dset = f.create_dataset("x_test_filename", (N,), "S20")
    
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
        print("Preprocess Training data {}/{}".format(checkpnt[i-1], N))
        filesname = files_name[checkpnt[i-1]:checkpnt[i]]
        with ThreadPoolExecutor(process_count) as pool:
            for img_array, file_name in tqdm(pool.map(preprocfun,
                                                      [(test_set_folder, file_name, img_resize)
                                                       for file_name in filesname]),
                                             total=checkpnt[i]-checkpnt[i-1], mininterval=1.0):
                x_test.append(img_array)
                x_test_filename.append(file_name)
            x_test_dset[checkpnt[i-1]:checkpnt[i],:,:,:] = x_test
            x_test_filename_dset[checkpnt[i-1]:checkpnt[i]] = x_test_filename
            x_test = []
            x_test_filename = []
    f.flush()
    f.close()
    print("Done.")
    return


if __name__ == "__main__":
    print('Test training Jpeg')
    train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file, train_tif_dir, _ = get_data_files_paths()
    img_resize = (64, 64)
    files_path, tags_list, labels_map = get_file_tag_list(train_csv_file, train_jpeg_dir)
    preprocess_train_data(files_path, tags_list, labels_map, 'test.h5', _train_transform_to_matrices, img_resize, 3)
#    x_train, y_train, y_map = preprocess_train_data(train_jpeg_dir, train_csv_file, img_resize)
#    plt.figure()
#    for i in range(20):
#        plt.subplot(4,5,i+1)
#        plt.imshow(x_train[i].squeeze())

    