"""
train.py
Train model.
"""
import os
import tensorflow as tf

from condgan import ConditionalGAN
from data import DataGenerator
from colors import colors_dict


PARAMS = {
    'dataset': 'wikiart',
    'generator_model': 'cond_stylegan',
    'discriminator_model': 'discriminator_progressive',
    # 'source_dir': '/home/alice/lemotif/data/wikiart-abstract-crop-2/',
    'source_dir': '/home/alice/data/wikiart-abstract-crop-2/',
    'tag_dict': colors_dict,
    'n_classes': 18,
    'max_filters': 512,
    'shape': (256, 256),
    'batch_size': 8,
    'z_dim': 256,
    'img_size': 256,
    'start_size': 64,
    'n_channels': 3,
    'iterations': {32: 250000, 64: 500000, 128: 500000, 256: 1000000, 512: 1000000},  # Number of steps
    'lr': {32: 0.00001, 64: 0.00001, 128: 0.00001, 256: 0.000001, 512: 0.00001},  # Learning rate
    'b1': 0.5,  # Adam beta1
    'b2': 0.99,   # Adam beta2
    'loss_weights': {'class_label': 1000, 'gen_label': 1000},
    'save_int': 10000,  # Number of epochs before generating output
    'save_dir': '/home/alice/lemotif/outputs/wikiart_residuals_acgan_sg_2/',  # Output data directory
    'verbose': True
}

if __name__ == '__main__':
    if not os.path.isdir(PARAMS['save_dir']): os.mkdir(PARAMS['save_dir'])
    f = open(os.path.join(PARAMS['save_dir'], "args.txt"), "w")
    f.write(str(PARAMS))
    f.close()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.compat.v1.enable_eager_execution(config=config)
    with tf.device("/cpu:0"):
        dataset = DataGenerator(**PARAMS).gen_dataset(parser='parse_cond')
    model = ConditionalGAN(**PARAMS)
    model.train(train_data=dataset, **PARAMS)
