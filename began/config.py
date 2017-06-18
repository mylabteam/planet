# coding: utf-8
import os

THIS_DIR = os.path.dirname(__file__)


class BEGANConfig:
    def __init__(self):
        self.project_dir = os.path.join(THIS_DIR,'')
        self.h5_train_file = os.path.join(self.project_dir,'..','results','train_jpg_rgb.h5')
        self.h5_test_file = os.path.join(self.project_dir,'..','results','test_jpg_rgb.h5')
        self.resource_dir = os.path.join(self.project_dir,'resource')
        self.dataset_dir = self.resource_dir
        self.generated_dir = os.path.join(self.project_dir,'generated')
        self.dataset_filename = os.path.join(self.dataset_dir,'dataset.pkl')
        self.image_width = 64
        self.image_height = 64
        self.n_filters = 128
        self.n_layer_in_conv = 2
        self.hidden_size = 64
        self.initial_k = 0
        self.gamma = 0.5
        self.lambda_k = 0.001
        self.batch_size = 16
        self.initial_lr = 0.0001
        self.min_lr = 0.00001
        self.lr_decay_rate = 0.9

        self.autoencoder_weight_filename = os.path.join(self.dataset_dir,'autoencoder.hd5')
        self.generator_weight_filename = os.path.join(self.dataset_dir,'generator.hd5')
        self.discriminator_weight_filename = os.path.join(self.dataset_dir,'discriminator.hd5')
        self.training_log = os.path.join(self.generated_dir,'training_log.csv')
        self.training_graph = os.path.join(self.generated_dir,'training.png')
