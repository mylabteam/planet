import numpy as np
import json
from sklearn.metrics import fbeta_score
#from sklearn.model_selection import train_test_split
from keras.utils.io_utils import HDF5Matrix
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import tensorflow.contrib.keras.api.keras as k
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import Callback, EarlyStopping
from tensorflow.contrib.keras import backend


class LossHistory(Callback):
    def __init__(self):
        super(LossHistory,self).__init__()
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


class AmazonKerasClassifier:
    def __init__(self):
        self.losses = []
        self.classifier = Sequential()

    def add_conv_layer(self, img_size=(32, 32), img_channels=3):
        self.classifier.add(BatchNormalization(input_shape=(img_size[0], img_size[1], img_channels)))

        self.classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(32, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(64, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(128, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(256, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))


    def add_flatten_layer(self):
        self.classifier.add(Flatten())


    def add_ann_layer(self, output_size):
        self.classifier.add(Dense(512, activation='relu'))
        self.classifier.add(BatchNormalization())
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(output_size, activation='sigmoid'))
        print(self.classifier.summary())

    def _get_fbeta_score(self, classifier, X_valid, y_valid):
        p_valid = classifier.predict(X_valid)
        if type(y_valid) is HDF5Matrix:
            return fbeta_score(np.array(y_valid), np.array(p_valid) > 0.2, beta=2, average='samples')
        else:
            return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')

    def train_model(self, x_train, y_train, x_valid, y_valid, learn_rate=0.001, epoch=5, batch_size=128, train_callbacks=()):
        history = LossHistory()

        opt = Adam(lr=learn_rate)

        self.classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


        # early stopping will auto-stop training process if model stops learning after 3 epochs
        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=2, mode='auto')
        
#        datagen = ImageDataGenerator(
#                width_shift_range=0.1,
#                height_shift_range=0.1,
#                fill_mode="reflect",
#                horizontal_flip=True,
#                vertical_flip=True)
#
#        # fits the model on batches with real-time data augmentation:
#        self.classifier.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
#                            steps_per_epoch=len(x_train) / batch_size,
#                            epochs=epoch,
#                            verbose=2,
#                            validation_data=(x_valid, y_valid),
#                            callbacks=[history] + train_callbacks + [earlyStopping])
                
        # Fix AttributeError following https://github.com/fchollet/keras/pull/6502/files
        #self.classifier.fit(x_train, y_train, shuffle="batch", batch_size=batch_size)
        self.classifier.fit(x_train, y_train,
                            shuffle="batch",
                            batch_size=batch_size,
                            epochs=epoch,
                            verbose=2,
                            validation_data=(x_valid, y_valid),
                            callbacks=[history] + train_callbacks + [earlyStopping])
        fbeta_score = self._get_fbeta_score(self.classifier, x_valid, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]
#    def train_model(self, x_train, y_train, learn_rate=0.001, epoch=5, batch_size=128, validation_split_size=0.2, train_callbacks=()):
#        history = LossHistory()
#
#        X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train,
#                                                              test_size=validation_split_size)
#
#        opt = Adam(lr=learn_rate)
#
#        self.classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#
#
#        # early stopping will auto-stop training process if model stops learning after 3 epochs
#        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
#
#        self.classifier.fit(X_train, y_train,
#                            batch_size=batch_size,
#                            epochs=epoch,
#                            verbose=1,
#                            validation_data=(X_valid, y_valid),
#                            callbacks=[history] + train_callbacks + [earlyStopping])
#        fbeta_score = self._get_fbeta_score(self.classifier, X_valid, y_valid)
#        return [history.train_losses, history.val_losses, fbeta_score]

    def save_weights(self, weight_file_path):
        self.classifier.save_weights(weight_file_path)
    
    def save_model(self, model_file_path):
        json_string = self.classifier.to_json()
        with open(model_file_path, 'w') as f:
            json.dump(json_string, f)
        
    def load_weights(self, weight_file_path):
        self.classifier.load_weights(weight_file_path)

    def load_model(self, model_file_path):
        with open(model_file_path, 'r') as f:
            json_string = json.load(f)
            self.classifier = model_from_json(json_string)

    def predict(self, x_test):
        predictions = self.classifier.predict(x_test)
        return predictions

    def map_predictions(self, predictions, labels_map, thresholds):
        """
        Return the predictions mapped to their labels
        :param predictions: the predictions from the predict() method
        :param labels_map: the map
        :param thresholds: The threshold of each class to be considered as existing or not existing
        :return: the predictions list mapped to their labels
        """
        predictions_labels = []
        for prediction in predictions:
            labels = [labels_map[i] for i, value in enumerate(prediction) if value > thresholds[i]]
            predictions_labels.append(labels)

        return predictions_labels

    def close(self):
        backend.clear_session()


if __name__ == "__main__":
    import h5py
    import matplotlib.pyplot as plt
    validation_split_size = 0.2
    h5_train_file = "results/train_jpg_rgb.h5"
    h5_test_file = "results/test_jpg_rgb.h5"
    
    with h5py.File(h5_train_file, "r") as f:
        N_train = f["x_train"].shape[0]
        my_array = f["y_map"][()].tolist()
        y_map = {int(key):value for key, value in [tuple(x.split("=")) for x in my_array]}
    
    N_split = int(round(N_train * (1-validation_split_size)))
    
    x_train = HDF5Matrix(h5_train_file, "x_train", start=0, end=N_split)
    y_train = HDF5Matrix(h5_train_file, "y_train", start=0, end=N_split)
    # %%
    X = x_train[10500]
    plt.figure()
    plt.subplot(4,5,1)
    plt.imshow(X)
    plt.title('original')
    
    datagen = ImageDataGenerator(
                rotation_range=90,
                width_shift_range=0.1,
                height_shift_range=0.1,
                fill_mode="reflect",
                horizontal_flip=True,
                vertical_flip=True)
    i = 2
    for batch in datagen.flow(X.reshape(1,64,64,3), batch_size=1):
        plt.subplot(4,5,i)
        plt.imshow(batch.squeeze())
        i += 1
        if i > 20:
            break