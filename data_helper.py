import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import skimage
from skimage import io
from skimage.transform import resize
from itertools import chain
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

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


def _get_tif_image(img, img_resize, meanstd):
    img = img.astype('float32')
    img = img[:,:,:3]
    #skimage.img_as_float(img)
    for i in range(3):
        img[:,:,i] = img[:,:,i]-np.mean(img[:,:,i])
        # Normalize variance
        img[:,:,i] = img[:,:,i]/np.std(img[:,:,i])
        # Scale to reference 
        img[:,:,i] = img[:,:,i]*meanstd[i,1] + meanstd[i,0]
        # Clip any values going out of the valid range
        img[:,:,i] = np.clip(img[:,:,i],0,1)
    #img[:,:,3] = (img[:,:,3]-9571.0479)/2297.3435
    #img[:,:,3] = np.clip(img[:,:,3],0,1)
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
    file_path, tags, labels_map, img_resize, ref_mustd = list(args[0])
    
    img = io.imread(file_path)
    # Augment the image `img` here
    if file_path[-3:] == 'tif':
        key = file_path.split("/")[-1][:-4]
        img_array = _get_tif_image(img, img_resize, ref_mustd[key])
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
    test_set_folder, file_name, img_resize, ref_mustd = list(args[0])
    file_path = "{}/{}".format(test_set_folder, file_name)
    img = io.imread(file_path)
    # Augment the image `img` here
    if file_path[-3:] == 'tif':
        key = file_path.split("/")[-1][:-4]
        img_array = _get_tif_image(img, img_resize, ref_mustd[key])
    else:
        img_array = _get_image(img, img_resize)
    return img_array, file_name


def get_dict_means_stds(jpg_path, process_count=cpu_count()):
    jpg_mean_std = {}
    with ThreadPoolExecutor(process_count) as pool:
        for key, value in tqdm(pool.map(get_ref_means_stds,
                                        [filepath for filepath in jpg_path]),
                                        total=len(jpg_path), mininterval=1.0):
            jpg_mean_std[key] = value
    return jpg_mean_std


def get_ref_means_stds(filepath):
    key = filepath.split("/")[-1][:-4]
    img = io.imread(filepath)
    # convert to double
    img = skimage.img_as_float(img)
    rgbmean = img.reshape((-1,3)).mean(axis=0)
    rgbstd = img.reshape((-1,3)).std(axis=0)
    value = np.concatenate((rgbmean[:,np.newaxis], rgbstd[:,np.newaxis]), axis=1)
    return key, value


def max_nir(filepath):
    img = io.imread(filepath)
    img = img.astype('float32')
    return img[:,:,3].max()


if __name__ == "__main__":
    print('Test training Jpeg')
    train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file, train_tif_dir, test_tif_dir = get_data_files_paths()
    img_resize = (64, 64)
    tif_path, tags_list, labels_map = get_file_tag_list(train_csv_file, train_tif_dir, ext='tif')
    
    nir = []
    with ThreadPoolExecutor(cpu_count()) as pool:
        for val in tqdm(pool.map(max_nir, [filepath for filepath in tif_path]),
                                        total=len(tif_path), mininterval=1.0):
            nir.append(val)
    print(np.mean(nir))
    #>>> np.mean(nir)
    #9571.05
    #>>> np.std(nir)
    #2297.3435
    
#    jpg_path, _, _ = get_file_tag_list(train_csv_file, train_jpeg_dir, ext='jpg')
#    
#    jpg_mean_std = {}
#    with ThreadPoolExecutor(cpu_count()) as pool:
#        for key, value in tqdm(pool.map(get_ref_means_stds,[filepath for filepath in jpg_path]),
#                                        total=len(jpg_path), mininterval=1.0):
#            jpg_mean_std[key] = value
#    
#    n = 1
#    img1 = io.imread(tif_path[n])
#    key = tif_path[n].split("/")[-1][:-4]
#    jpg_mean_std[key]
#    hehe = _get_tif_image(img1, (64,64), jpg_mean_std[key])
#    
#    img2 = io.imread(jpg_path[n])
#    plt.figure()
#    plt.subplot(121)
#    plt.imshow(hehe[:,:,:3])
#    plt.subplot(122)
#    plt.imshow(resize(img2, (64,64)))
#    
#    
#    jpg_path_test = ["{}/{}".format(test_jpeg_dir, filename) for filename in os.listdir(test_jpeg_dir)] + \
#                ["{}/{}".format(test_jpeg_additional, filename) for filename in os.listdir(test_jpeg_additional)]
#    jpg_test_mean_std = {}
#    with ThreadPoolExecutor(cpu_count()) as pool:
#        for key, value in tqdm(pool.map(get_ref_means_stds,[filepath for filepath in jpg_path_test]),
#                                        total=len(jpg_path_test), mininterval=1.0):
#            jpg_test_mean_std[key] = value
#    
#    preprocess_train_data(files_path, tags_list, labels_map, 'test.h5', _train_transform_to_matrices, img_resize, 3)
#    x_train, y_train, y_map = preprocess_train_data(train_jpeg_dir, train_csv_file, img_resize)
#    plt.figure()
#    for i in range(20):
#        plt.subplot(4,5,i+1)
#        plt.imshow(x_train[i].squeeze())

    