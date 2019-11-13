"""
train.py
Train model.
"""

import tensorflow as tf

from model import ConditionalGAN
from data import DataGenerator
from colors import colors_dict


PARAMS = {
    'dataset': 'wikiart',
    'source_dir': '/content/drive/My Drive/Colab Data/lemotif/data/wikiart-abstract-crop-2/',
    'tag_dict': colors_dict,
    'n_classes': 18,
    'shape': (128, 128),
    'batch_size': 8,
    'z_dim': 256,
    'img_size': 128,
    'start_size': 32,
    'iterations': 100000,  # Number of epochs
    'lr': 0.000001,  # Learning rate
    'b1': 0.5,  # Adam beta1
    'b2': 0.99,   # Adam beta2
    'save_int': 5000,  # Number of epochs before generating output
    'save_dir': '/content/drive/My Drive/Colab Data/lemotif/outputs/wikiart-abstract/',  # Output data directory
    'verbose': True
}

if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    with tf.device("/cpu:0"):
        dataset = DataGenerator(**PARAMS).gen_dataset()
    model = ConditionalGAN(**PARAMS)
    model.train(train_data=dataset, **PARAMS)
