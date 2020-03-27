"""
condgan.py
Conditional GAN model
"""
import glob
import os

import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from tqdm import tqdm

import models


class ConditionalGAN():

    def __init__(self,
                 batch_size,
                 z_dim,
                 n_classes,
                 generator_model,
                 discriminator_model,
                 loss_weights,
                 max_filters,
                 n_channels=3,
                 map_layers=4,
                 conv_init='he_normal',
                 disc_iterations=1,
                 gen_iterations=1,
                 save_checkpts=True,
                 verbose=True,
                 **kwargs):
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.num_classes = n_classes
        self.n_channels = n_channels
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self.loss_weights = loss_weights
        self.max_filters = max_filters
        self.map_layers = map_layers
        self.conv_init = conv_init
        self.disc_iterations = disc_iterations
        self.gen_iterations = gen_iterations
        self.save_checkpts = save_checkpts
        self.verbose = verbose
        self.best_gen_loss = None
        self.best_disc_loss = None
        self.gen_loss_curve = []
        self.disc_loss_curve = []


    def init_models(self, start_size, fade_on, lr, b1, b2):
        # Build models
        generator_model = getattr(models, self.generator_model)
        self.generator = generator_model(start_size, fade_on, self.z_dim,
                                         num_classes=self.num_classes,
                                         conv_init=self.conv_init,
                                         batch_size=self.batch_size,
                                         n_channels=self.n_channels,
                                         start_filters=self.max_filters)
        discriminator_model = getattr(models, self.discriminator_model)
        self.discriminator = discriminator_model(start_size, fade_on,
                                                 num_classes=self.num_classes,
                                                 conv_init=self.conv_init,
                                                 batch_size=self.batch_size,
                                                 n_channels=self.n_channels,
                                                 max_filters=self.max_filters)

        if self.verbose:
            print(self.generator.summary())
            print(self.discriminator.summary())

        if isinstance(lr, dict):
            lr = lr[start_size]

        self.gen_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2)
        self.disc_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr*4, beta1=b1, beta2=b2)

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.gen_optimizer,
                                              discriminator_optimizer=self.disc_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    def generator_pass(self, noise, conditioning_noised, fade, ones, training=True):
        if 'stylegan' in self.generator_model:
            generated_imgs = self.generator(inputs=[noise, conditioning_noised, fade, ones], training=training)
        else:
            generated_imgs = self.generator(inputs=[noise, conditioning_noised, fade], training=training)

        return generated_imgs

    def train_step(self, imgs, conditioning, fade, ones):
        # Generate noise from normal distribution
        noise = tf.random.normal([self.batch_size, self.z_dim])
        conditioning_noised = tf.clip_by_value(
            conditioning + tf.random.normal([self.batch_size, self.num_classes], stddev=.1), 0, 1.2)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_imgs = self.generator_pass(noise, conditioning_noised, fade, ones, training=True)

            real_disc, real_labels_pred = \
                self.discriminator(inputs=[imgs, conditioning_noised, fade], training=True)
            generated_disc, generated_labels_pred = \
                self.discriminator(inputs=[generated_imgs, conditioning_noised, fade], training=True)

            gen_loss = self.generator_loss(generated_disc)
            disc_loss = self.discriminator_loss(imgs, generated_imgs, real_disc, generated_disc, conditioning, fade)
            self.tot_disc_loss = disc_loss
            self.tot_gen_loss = gen_loss
            if self.loss_weights['class_label'] or self.loss_weights['gen_label']:
                self.true_label_loss, self.gen_label_loss = self.conditioning_loss(
                    conditioning_noised, real_labels_pred, generated_labels_pred)
                self.tot_disc_loss += self.true_label_loss * self.loss_weights['class_label']
                self.tot_gen_loss += self.gen_label_loss * self.loss_weights['gen_label']

            self.gen_loss_curve.append(self.tot_gen_loss)
            self.disc_loss_curve.append(self.tot_disc_loss)

        gradients_of_generator = gen_tape.gradient(self.tot_gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(self.tot_disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        self.best_gen_loss = min(self.best_gen_loss, self.tot_gen_loss) if self.best_gen_loss else self.tot_gen_loss
        self.best_disc_loss = min(self.best_disc_loss, self.tot_disc_loss) if self.best_disc_loss else self.tot_disc_loss

    def train(self, train_data, img_size, start_size, iterations, lr, save_dir, b1, b2, save_int, **kwargs):
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        gen_steps, gen_res, disc_steps, disc_res = self.load_saved_models(save_dir)

        if gen_steps:
            loop_start_size = gen_res
            fade_on = True
        else:
            loop_start_size = start_size
            fade_on = False
        self.init_models(loop_start_size, True, lr, b1, b2)

        # Number of progressive resolution stages
        resolutions = int(np.log2(img_size/loop_start_size)) + 1
        ones = tf.cast(tf.ones((self.batch_size, 1)), tf.float32)

        for resolution in range(resolutions):
            if resolution > 0: fade_on = True
            print('Resolution: ', loop_start_size*2**resolution)
            res_iterations = iterations[loop_start_size*2**resolution]
            stage_iterations = max(0, res_iterations - gen_steps) if resolution == 0 else res_iterations
            progress = tqdm(train_data.take(stage_iterations))
            for iteration, (imgs, conditioning) in enumerate(progress):
                # imgs, conditioning = sample['image'], sample['label']
                if resolution < resolutions - 1:
                    imgs = tf.image.resize_images(imgs, (loop_start_size*2**resolution, loop_start_size*2**resolution),
                                                  align_corners=True)
                if len(conditioning.get_shape().as_list()) < 2:
                    conditioning = tf.one_hot(conditioning, self.num_classes, axis=-1)
                fade = min(iteration/(stage_iterations//2.0), 1.0) if (resolution > 0 and fade_on) else 1.0
                fade_tensor = tf.constant(fade, shape=(self.batch_size, 1), dtype=tf.float32)
                self.train_step(imgs=imgs, conditioning=conditioning, fade=fade_tensor, ones=ones)
                progress.set_postfix(best_gen_loss=self.best_gen_loss.numpy(), best_disc_loss=self.best_disc_loss.numpy(),
                                     gen_loss=self.tot_gen_loss.numpy(), disc_loss=self.tot_disc_loss.numpy())

                # Save every n intervals
                if iteration % save_int == 0:
                    random_classes = []
                    for i in range(self.num_classes):
                        random_label = [0] * self.num_classes
                        random_label[i] = 1.0
                        random_classes.append(random_label)
                    conditioning = tf.stack(random_classes)
                    self.generate(iteration + 1, save_dir, self.num_classes, conditioning, fade)
                    self.save_learning_curve(save_dir, loop_start_size*2**resolution)
                    self.save_models(save_dir, loop_start_size*2**resolution, iteration, gen_steps, disc_steps)

            if resolution < resolutions - 1:
                print('Updating models to add new layers for next resolution.')
                self.update_models(loop_start_size*2**resolution)


    def update_models(self, size):
        # Updates generator and discriminator models to add new layers corresponding to next resolution size
        # Retains weights previously learned from lower resolutions
        new_size = size*2
        generator_model = getattr(models, self.generator_model)
        new_generator = generator_model(new_size, True, self.z_dim,
                                        num_classes=self.num_classes,
                                        conv_init=self.conv_init,
                                        batch_size=self.batch_size,
                                        n_channels=self.n_channels,
                                        start_filters=self.max_filters)
        discriminator_model = getattr(models, self.discriminator_model)
        new_discriminator = discriminator_model(new_size, True,
                                                num_classes=self.num_classes,
                                                conv_init=self.conv_init,
                                                batch_size=self.batch_size,
                                                n_channels=self.n_channels,
                                                max_filters=self.max_filters)
        new_gen_layers = [layer.name for layer in new_generator.layers]
        new_disc_layers = [layer.name for layer in new_discriminator.layers]
        for layer in self.generator.layers:
            if ('dense' in layer.name or 'conv' in layer.name or 'batch' in layer.name) and layer.name in new_gen_layers:
                print('Updating ', layer.name)
                new_generator.get_layer(layer.name).set_weights(self.generator.get_layer(layer.name).get_weights())
            else:
                print('No match with weights ', layer.name)
        for layer in self.discriminator.layers:
            if ('dense' in layer.name or 'conv' in layer.name or 'batch' in layer.name) and layer.name in new_disc_layers:
                print('Updating ', layer.name)
                new_discriminator.get_layer(layer.name).set_weights(self.discriminator.get_layer(layer.name).get_weights())
            else:
                print('No match with weights ', layer.name)
        self.generator, self.discriminator = new_generator, new_discriminator

        if self.verbose:
            print(self.generator.summary())
            print(self.discriminator.summary())

    def plot_models(self, size, save_dir):
        # Plot model structure
        tf.keras.utils.plot_model(self.generator, show_shapes=True,
                                  to_file=os.path.join(save_dir, str(size) + '_gen.jpg'))
        tf.keras.utils.plot_model(self.discriminator, show_shapes=True,
                                  to_file=os.path.join(save_dir, str(size) + '_disc.jpg'))

    def generator_loss(self, generated_disc):
        # WGAN-GP loss: https://arxiv.org/pdf/1704.00028.pdf
        # Negative so that gradient descent maximizes critic score received by generated output
        return -tf.reduce_mean(generated_disc)

    def discriminator_loss(self, real_imgs, generated_imgs, real_disc, generated_disc, conditioning, fade, gp_lambda=10,
                           epsilon=0.001):
        # WGAN-GP loss: https://arxiv.org/pdf/1704.00028.pdf
        # Difference between critic scores received by generated output vs real video
        # Lower values mean that the real video samples are receiving higher scores, therefore
        # gradient descent maximizes discriminator accuracy
        out_size = real_imgs.get_shape().as_list()
        d_cost = tf.reduce_mean(generated_disc) - tf.reduce_mean(real_disc)
        alpha = tf.random.uniform(
            shape=[self.batch_size, out_size[2], out_size[2], 3],
            minval=0.,
            maxval=1.
        )
        diff = generated_imgs - real_imgs
        # Real imgs adjusted by randomly weighted difference between real vs generated
        interpolates = real_imgs + (alpha * diff)
        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            interpolates_disc, _ = self.discriminator([interpolates, conditioning, fade], training=False)
        # Gradient of critic score wrt interpolated imgs
        gradients = tape.gradient(interpolates_disc, [interpolates])[0]
        # Euclidean norm of gradient for each sample
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        # Gradient norm penalty is the average distance from 1
        gradient_penalty = tf.reduce_mean((norm - 1.) ** 2) * gp_lambda
        epsilon_penalty = tf.reduce_mean(real_disc) * epsilon

        return d_cost + gradient_penalty + epsilon_penalty

    def conditioning_loss(self, true_labels, true_pred, gen_pred):
        ce_true = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_labels, logits=true_pred))
        ce_gen = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_labels, logits=gen_pred))

        return ce_true, ce_gen

    def generate(self, epoch, save_dir, num_out, conditioning, fade=1.0):
        gen_noise = tf.random.normal([num_out, self.z_dim])
        fade, ones = tf.constant(fade, shape=[num_out, 1]), tf.ones((num_out, 1))
        generated_imgs = self.generator_pass(gen_noise, conditioning, fade, ones, training=False)

        self.save_img(generated_imgs, conditioning, save_dir,
                      name=str(generated_imgs[0].shape[-2]) + '_' + str(epoch) + '_')

    def save_img(self, img_tensor, conditioning, save_dir, name):
        img = tf.cast(255 * (img_tensor + 1)/2, tf.uint8)
        for i, ind_img in enumerate(img):
            encoded = tf.image.encode_jpeg(ind_img)
            tf.write_file(os.path.join(save_dir, name + str(i) + '.jpg'), encoded)
        np.savetxt(os.path.join(save_dir, name + '.txt'), conditioning)

    def save_learning_curve(self, save_dir, res):
        plt.plot(range(len(self.gen_loss_curve)), self.gen_loss_curve, color='g', linewidth='1')
        plt.xlabel("Iterations")
        plt.ylabel("Generator Loss")
        plt.savefig(os.path.join(save_dir, 'gen_loss_' + str(res) + '.jpg'), bbox_inches='tight')
        plt.clf()

        plt.plot(range(len(self.disc_loss_curve)), self.disc_loss_curve, color='g', linewidth='1')
        plt.xlabel("Iterations")
        plt.ylabel("Discriminator Loss")
        plt.savefig(os.path.join(save_dir, 'disc_loss_' + str(res) + '.jpg'), bbox_inches='tight')
        plt.clf()

    def load_saved_models(self, save_dir):
        gen_checkpoints = glob.glob(os.path.join(save_dir, 'gen-*.h5'))
        max_gen, gen_res, max_disc, disc_res = 0, None, 0, None
        if len(gen_checkpoints) > 0:
            gen_res = max([int(os.path.basename(path).split('.')[0].split('-')[1]) for path in gen_checkpoints])
            gen_checkpoints_max = glob.glob(os.path.join(save_dir, 'gen-' + str(gen_res) + '-*.h5'))
            max_steps = [int(os.path.basename(path).split('.')[0].split('-')[2]) for path in gen_checkpoints_max]
            max_gen = max(max_steps)
            self.generator.load_weights(
                os.path.join(save_dir, 'gen-' + str(gen_res) + '-' + str(max_gen) + '.h5'), by_name=True)

        disc_checkpoints = glob.glob(os.path.join(save_dir, 'disc-*.h5'))
        if len(disc_checkpoints) > 0:
            disc_res = max([int(os.path.basename(path).split('.')[0].split('-')[1]) for path in disc_checkpoints])
            disc_checkpoints_max = glob.glob(os.path.join(save_dir, 'disc-' + str(disc_res) + '-*.h5'))
            max_steps = [int(os.path.basename(path).split('.')[0].split('-')[2]) for path in disc_checkpoints_max]
            max_disc = max(max_steps)
            self.discriminator.load_weights(
                os.path.join(save_dir, 'disc-' + str(disc_res) + '-' + str(max_disc) + '.h5'), by_name=True)

        return max_gen, gen_res, max_disc, disc_res

    def save_models(self, save_dir, res, current_steps, gen_steps, disc_steps):
        self.generator.save(
            os.path.join(save_dir, 'gen-' + str(res) + '-' + str(gen_steps + current_steps) + '.h5'))
        self.discriminator.save(
            os.path.join(save_dir, 'disc-' + str(res) + '-' + str(disc_steps + current_steps) + '.h5'))