"""
train.py
Train model.
"""

import tensorflow as tf

from condstylegan.model import ConditionalStyleGAN
from condstylegan.data import DataGenerator

PARAMS = {
    'dataset': 'mirflickr25k',
    'shape': (256, 256),
    'batch_size': 8,
    'iterations': 100000,  # Number of epochs
    'lr': 0.00001,  # Learning rate
    'b1': 0.5,  # Adam beta1
    'b2': 0.99,   # Adam beta2
    'save_int': 1000,  # Number of epochs before generating output
    'save_dir': '/content/drive/My Drive/Colab Data/lemotif/outputs',  # Output data directory
    'verbose': True
}

if PARAMS['dataset'] == 'mirflickr25k':
    PARAMS['source_dir'] =  '/content/drive/My Drive/Colab Data/lemotif/mirflickr25k/imgs'
    PARAMS['annot_dir'] = '/content/drive/My Drive/Colab Data/lemotif/mirflickr25k/annotations'
    PARAMS['tag_dict'] = {'god':
                              {'annotation': ['sky'],
                               'annotation_dir': '/content/drive/My Drive/Colab Data/lemotif/mirflickr25k/annotations',
                               'user_tag': [],
                               'user_tag_dir': None}
                          }

if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    dataset = DataGenerator(**PARAMS).gen_dataset()
    model = ConditionalStyleGAN(**PARAMS)
    model.train(data=dataset, **PARAMS)
