"""
autoencoder.py
Autoencoder-style image generation model.
"""
import glob
import os

import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

import models
from utils import gauss_kernel


class AutoEncoderGAN():

    def __init__(self,
                 batch_size,
                 z_dim,
                 enc_model,
                 dec_model,
                 disc_model,
                 loss_weights,
                 max_filters,
                 conv_init='he_normal',
                 verbose=True,
                 **kwargs):
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.enc_model = enc_model
        self.dec_model = dec_model
        self.disc_model = disc_model
        self.loss_weights = loss_weights
        self.max_filters = max_filters
        self.conv_init = conv_init
        self.verbose = verbose
        self.best_gen_loss = None
        self.best_disc_loss = None
        self.gen_loss_curve = []
        self.disc_loss_curve = []

        if 'blur_kernel' in kwargs.keys() and kwargs['blur_kernel']:
            self.gauss_kernel = gauss_kernel(kwargs['blur_kernel'])

    def init_models(self, in_size, out_size, disc_size, lr, b1, b2):
        """Define encoder, decoder, and discriminator models if using plus optimizers."""
        encoder_model = getattr(models, self.enc_model)
        decoder_model = getattr(models, self.dec_model)

        self.encoder = encoder_model(in_size, self.z_dim, max_filters=self.max_filters, conv_init=self.conv_init)
        self.decoder = decoder_model(out_size, self.z_dim, max_filters=self.max_filters, conv_init=self.conv_init)
        if self.loss_weights['discriminator'] > 0 and self.disc_model:
            discriminator_model = getattr(models, self.disc_model)
            self.discriminator = discriminator_model(disc_size, max_filters=self.max_filters, conv_init=self.conv_init)

        if self.verbose:
            print(self.encoder.summary())
            print(self.decoder.summary())
            if self.loss_weights['discriminator'] > 0:
                print(self.discriminator.summary())

        self.gen_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2)
        if self.loss_weights['discriminator'] > 0:
            self.disc_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr*4, beta1=b1, beta2=b2)

    def train_step(self, img_in, true_img, out_size, disc_size):
        """Make one update step."""
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_z = self.encoder(img_in, training=True)
            gen_img = self.decoder(gen_z, training=True)

            self.tot_gen_loss = 0
            if self.loss_weights['discriminator'] > 0:
                if disc_size < out_size:
                    true_img_disc = tf.random_crop(true_img, [self.batch_size, disc_size, disc_size, 3])
                    gen_img_disc = tf.random_crop(gen_img, [self.batch_size, disc_size, disc_size, 3])
                else:
                    true_img_disc = true_img
                    gen_img_disc = gen_img

                real_disc = self.discriminator(true_img_disc, training=True)
                generated_disc = self.discriminator(gen_img_disc, training=True)

                gen_loss = self.generator_loss(generated_disc) * self.loss_weights['discriminator']
                disc_loss = self.discriminator_loss(true_img_disc, gen_img_disc, real_disc, generated_disc) * \
                            self.loss_weights['discriminator']
                self.tot_disc_loss = disc_loss
                self.tot_gen_loss += gen_loss
                self.disc_loss_curve.append(self.tot_disc_loss)

            l2_loss = self.l2_loss(gen_img, true_img) * self.loss_weights['l2_gen']
            self.tot_gen_loss += l2_loss
        self.gen_loss_curve.append(self.tot_gen_loss)

        gradients_gen = gen_tape.gradient(
            self.tot_gen_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(gradients_gen, self.encoder.trainable_variables + self.decoder.trainable_variables))
        self.best_gen_loss = min(self.best_gen_loss, self.tot_gen_loss) if self.best_gen_loss else self.tot_gen_loss

        if self.loss_weights['discriminator'] > 0:
            gradients_disc = disc_tape.gradient(self.tot_disc_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(gradients_disc, self.discriminator.trainable_variables))
            self.best_disc_loss = min(self.best_disc_loss, self.tot_disc_loss) if self.best_disc_loss else self.tot_disc_loss

            return (gen_loss, disc_loss, l2_loss)
        else:
            return (l2_loss)

    def train(self, train_data, in_size, out_size, disc_size, iterations, lr, save_dir, b1, b2, save_int, **kwargs):
        """Train over specified number of iterations per settings."""
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        self.init_models(in_size, out_size, disc_size, lr, b1, b2)
        gen_steps, disc_steps = self.load_saved_models(save_dir)

        source_imgs = None
        progress = tqdm(train_data.take(iterations))
        for iteration, true_img in enumerate(progress):
            if kwargs['blur_kernel']:
                proc_img = tf.nn.depthwise_conv2d(true_img, self.gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
            else:
                proc_img = true_img
            if in_size < out_size:
                proc_img = tf.image.resize(proc_img, (in_size, in_size), align_corners=True)
            if source_imgs is None:
                source_imgs = proc_img
                self.save_img(source_imgs, save_dir, name='imgs_in_')
            losses = self.train_step(img_in=proc_img, true_img=true_img, out_size=out_size, disc_size=disc_size)
            if self.loss_weights['discriminator'] > 0:
                progress.set_postfix(best_gen_loss=self.best_gen_loss.numpy(), best_disc_loss=self.best_disc_loss.numpy(),
                                     gen_loss=losses[0].numpy(), disc_loss=losses[1].numpy(), l2_loss=losses[2].numpy())
            else:
                progress.set_postfix(l2_loss=losses.numpy())

            if iteration % save_int == 0 or iteration == iterations - 1:
                self.generate(source_imgs, iteration + 1 + gen_steps, save_dir)
                self.save_learning_curve(save_dir)
                self.save_models(save_dir, iteration, gen_steps, disc_steps)


    def generator_loss(self, generated_disc):
        """Gen loss from WGAN-GP loss: https://arxiv.org/pdf/1704.00028.pdf"""
        # Negative so that gradient descent maximizes critic score received by generated output
        return -tf.reduce_mean(generated_disc)

    def discriminator_loss(self, real_imgs, generated_imgs, real_disc, generated_disc, gp_lambda=10, epsilon=0.001):
        """Disc loss from WGAN-GP loss: https://arxiv.org/pdf/1704.00028.pdf"""
        # Difference between critic scores received by generated output vs real image
        # Lower values mean that the real image samples are receiving higher scores, therefore
        # gradient descent maximizes discriminator accuracy
        out_size = real_imgs.get_shape().as_list()
        d_cost = tf.reduce_mean(generated_disc) - tf.reduce_mean(real_disc)
        alpha = tf.random.uniform(
            shape=[self.batch_size, out_size[2], out_size[2], 3],
            minval=0.,
            maxval=1.
        )
        diff = generated_imgs - real_imgs
        interpolates = real_imgs + (alpha * diff)
        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            interpolates_disc = self.discriminator([interpolates], training=False)
        # Gradient of critic score wrt interpolated imgs
        gradients = tape.gradient(interpolates_disc, [interpolates])[0]
        # Euclidean norm of gradient for each sample
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        # Gradient norm penalty is the average distance from 1
        gradient_penalty = tf.reduce_mean((norm - 1.) ** 2) * gp_lambda
        epsilon_penalty = tf.reduce_mean(real_disc) * epsilon

        return d_cost + gradient_penalty + epsilon_penalty

    def l2_loss(self, gen_img, true_img):
        """L2 loss."""
        if self.loss_weights['l2_loss_size'] != gen_img.get_shape().as_list()[1]:
            gen_img = tf.image.resize(gen_img,
                                      (self.loss_weights['l2_loss_size'], self.loss_weights['l2_loss_size']),
                                      align_corners=True)
        if self.loss_weights['l2_loss_size'] != true_img.get_shape().as_list()[1]:
            true_img = tf.image.resize(true_img,
                                       (self.loss_weights['l2_loss_size'], self.loss_weights['l2_loss_size']),
                                       align_corners=True)

        return tf.reduce_mean(tf.squared_difference(true_img, gen_img))

    def generate(self, source_imgs, epoch, save_dir):
        """Inference pass to produce and save image."""
        generated_imgs = self.decoder(self.encoder(source_imgs, training=False), training=False)

        self.save_img(generated_imgs, save_dir, name=str(generated_imgs[0].shape[-2]) + '_' + str(epoch) + '_')

    def save_img(self, img_tensor, save_dir, name):
        """Save out image as JPG."""
        img = tf.cast(255 * (img_tensor + 1)/2, tf.uint8)
        for i, ind_img in enumerate(img):
            encoded = tf.image.encode_jpeg(ind_img)
            tf.write_file(os.path.join(save_dir, name + str(i) + '.jpg'), encoded)

    def save_learning_curve(self, save_dir):
        """Save out losses as plot."""
        plt.plot(range(len(self.gen_loss_curve)), self.gen_loss_curve, color='g', linewidth='1')
        plt.xlabel("Iterations")
        plt.ylabel("Generator Loss")
        plt.savefig(os.path.join(save_dir, 'gen_loss.jpg'), bbox_inches='tight')
        plt.clf()

        if self.loss_weights['discriminator'] > 0:
            plt.plot(range(len(self.disc_loss_curve)), self.disc_loss_curve, color='g', linewidth='1')
            plt.xlabel("Iterations")
            plt.ylabel("Discriminator Loss")
            plt.savefig(os.path.join(save_dir, 'disc_loss.jpg'), bbox_inches='tight')
            plt.clf()

    def load_saved_models(self, save_dir):
        """Load models from checkpoints."""
        gen_checkpoints = glob.glob(os.path.join(save_dir, 'encoder-*.h5'))
        max_gen, max_disc = 0, 0
        if len(gen_checkpoints) > 0:
            all_steps = [int(os.path.basename(path).split('.')[0].split('-')[1]) for path in gen_checkpoints]
            max_gen = max(all_steps)
            self.encoder.load_weights(os.path.join(save_dir, 'encoder-' + str(max_gen) + '.h5'), by_name=True)
            self.decoder.load_weights(os.path.join(save_dir, 'decoder-' + str(max_gen) + '.h5'), by_name=True)
        if self.disc_model:
            disc_checkpoints = glob.glob(os.path.join(save_dir, 'disc-*.h5'))
            if len(disc_checkpoints) > 0:
                all_steps = [int(os.path.basename(path).split('.')[0].split('-')[1]) for path in disc_checkpoints]
                max_disc = max(all_steps)
                self.discriminator.load_weights(os.path.join(save_dir, 'disc-' + str(max_disc) + '.h5'), by_name=True)
        return max_gen, max_disc

    def save_models(self, save_dir, current_steps, gen_steps, disc_steps):
        """Save models as checkpoints."""
        self.encoder.save(os.path.join(save_dir, 'encoder-' + str(gen_steps + current_steps) + '.h5'))
        self.decoder.save(os.path.join(save_dir, 'decoder-' + str(gen_steps + current_steps) + '.h5'))
        if self.disc_model:
            self.discriminator.save(os.path.join(save_dir, 'disc-' + str(disc_steps + current_steps) + '.h5'))