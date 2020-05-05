"""
train_ae.py
Train AE-style model.
"""
import os
import tensorflow as tf

from autoencoder import AutoEncoderGAN
from data import DataGenerator
from colors import colors_dict


PARAMS = {
    'dataset': 'wikiart',
    'source_dir': '/home/alice/data/wikiart-abstract-crop/',
    'tag_dict': colors_dict,
    'in_size': 16,
    'out_size': 256,
    'disc_size': 256,
    'enc_model': 'encoder_residual',
    'dec_model': 'decoder_basic',
    'disc_model': 'discriminator_residual',
    'shape': (256, 256),
    'batch_size': 8,
    'z_dim': 512,
    'max_filters': 512,
    'iterations': 1000000,  # Number of steps
    'lr': 0.00001,  # Learning rate
    'b1': 0.5,  # Adam beta1
    'b2': 0.99,   # Adam beta2
    'loss_weights': {'l2_gen': 1000, 'l2_loss_size': 256, 'discriminator': 0.01},
    'blur_kernel': 0,
    'save_int': 20000,  # Number of steps before generating output
    'save_dir': '/home/alice/lemotif/outputs/wikiart_ae_nodisc_resid_16-256_blur0/',  # Output data directory
    'verbose': True
}

if __name__ == '__main__':
    if not os.path.isdir(PARAMS['save_dir']): os.mkdir(PARAMS['save_dir'])
    f = open(os.path.join(PARAMS['save_dir'], "args.txt"), "w")
    f.write(str(PARAMS))
    f.close()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    tf.compat.v1.enable_eager_execution(config=config)
    with tf.device("/cpu:0"):
        dataset = DataGenerator(**PARAMS).gen_dataset(parser='parse_blur')
    model = AutoEncoderGAN(**PARAMS)
    model.train(train_data=dataset, **PARAMS)
