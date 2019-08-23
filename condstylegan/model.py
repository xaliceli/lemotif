"""
model.py
Conditional StyleGAN model based on NVidia implementation
"""
import os
import random

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
from tqdm import tqdm

from utils import add_noise, instance_norm


class ConditionalStyleGAN():

    def __init__(self,
                 batch_size,
                 z_dim,
                 map_layers=4,
                 conv_init='he_normal',
                 disc_iterations=1,
                 gen_iterations=1,
                 save_checkpts=True,
                 verbose=True,
                 **kwargs):
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.num_classes = len(kwargs['tag_dict'].keys())
        self.map_layers = map_layers
        self.conv_init = conv_init
        self.disc_iterations = disc_iterations
        self.gen_iterations = gen_iterations
        self.save_checkpts = save_checkpts
        self.verbose = verbose
        self.best_gen_loss = None
        self.best_disc_loss = None
        self.fade = None

    def init_models(self, start_size, lr, b1, b2):
        # Build models
        self.generator = self.generator_model(start_size)
        self.discriminator = self.discriminator_model(start_size)

        if self.verbose:
            print(self.generator.summary())
            print(self.discriminator.summary())

        self.gen_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2)
        self.disc_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2)

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.gen_optimizer,
                                              discriminator_optimizer=self.disc_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    def train_step(self, imgs, conditioning, fade, ones, noise):
        # Generate noise from normal distribution
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_imgs = self.generator(inputs=[noise, conditioning, fade, ones], training=True)

            real_disc = self.discriminator(inputs=[imgs, conditioning, fade], training=True)
            generated_disc = self.discriminator(inputs=[generated_imgs, conditioning, fade], training=True)

            self.gen_loss = self.generator_loss(generated_disc)
            self.disc_loss = self.discriminator_loss(imgs, generated_imgs, real_disc, generated_disc, conditioning, fade)

        gradients_of_generator = gen_tape.gradient(self.gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(self.disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        self.best_gen_loss = min(self.best_gen_loss, self.gen_loss) if self.best_gen_loss else self.gen_loss
        self.best_disc_loss = min(self.best_disc_loss, self.disc_loss) if self.best_disc_loss else self.disc_loss

    def train(self, train_data, img_size, start_size, iterations, lr, save_dir, b1, b2, save_int, **kwargs):
        # Load from checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(save_dir)
        if latest_checkpoint:
            print('Loading checkpoint ' + latest_checkpoint)
            loop_start_size = start_size * 2**(int(os.path.basename(latest_checkpoint)[4])-1)
        else:
            loop_start_size = start_size
        self.init_models(loop_start_size, lr, b1, b2)
        if latest_checkpoint:
            self.checkpoint.restore(latest_checkpoint)

        # Number of progressive resolution stages
        resolutions = int(np.log2(img_size/loop_start_size)) + 1
        ones = tf.cast(tf.ones((self.batch_size, 1)), tf.float32)
        noise = tf.random_normal([self.batch_size, self.z_dim])

        for resolution in range(resolutions):
            print('Resolution: ', loop_start_size*2**resolution)
            self.fade = True if (resolution > 0 or loop_start_size > start_size) else False
            progress = tqdm(train_data.take(iterations))
            for iteration, (imgs, conditioning) in enumerate(progress):
                if resolution < resolutions - 1:
                    pool_factor = 2 ** (resolutions - resolution - 1)
                    imgs = kl.AveragePooling2D(pool_factor, padding='same')(imgs)
                fade = iteration/(iterations//2.0)
                if fade == 1:
                    self.fade = False
                imgs = tf.cast(imgs, tf.float32)
                conditioning = tf.cast(conditioning, tf.float32)
                fade = tf.constant(fade, shape=(self.batch_size, 1), dtype=tf.float32)
                self.train_step(imgs=imgs, conditioning=conditioning, fade=fade, ones=ones, noise=noise)
                progress.set_postfix(best_gen_loss=self.best_gen_loss.numpy(), best_disc_loss=self.best_disc_loss.numpy(),
                                     gen_loss=self.gen_loss.numpy(), disc_loss=self.disc_loss.numpy())

                # Save every n intervals
                if (iteration + 1) % save_int == 0:
                    random_classes = []
                    for i in range(self.batch_size):
                        random_label = [0] * self.num_classes
                        random_label[random.randint(0, self.num_classes - 1)] = 1
                        random_classes.append(random_label)
                    conditioning = tf.cast(tf.stack(random_classes), tf.float32)
                    self.generate(iteration + 1, save_dir, self.batch_size, conditioning)
                    if self.save_checkpts:
                        self.checkpoint.save(
                            file_prefix=os.path.join(save_dir,
                                                     "ckpt" + str(resolution+np.log2(loop_start_size/start_size)+1)))

            if resolution < resolutions - 1:
                print('Updating models to add new layers for next resolution.')
                self.update_models(loop_start_size*2**resolution)


    def generator_model(self, out_size, start_size=8, start_filters=512):

        # Fading function
        def blend_resolutions(upper, lower, alpha):
            upper = tf.multiply(upper, alpha)
            lower = tf.multiply(lower, tf.subtract(1, alpha))
            return kl.Add()([upper, lower])

        # For now we start at 8x8 and upsample by 2x each time
        conv_loop = int(np.log2(out_size/start_size)) + 2

        z = kl.Input(shape=(self.z_dim,))
        fade = kl.Input(shape=(1,))
        conditioning = kl.Input(shape=(self.num_classes,))
        constant = kl.Input(shape=(1,))

        # Mapping network
        w = tf.concat((z, conditioning), 1) # Concatenate noise input with condition labels
        for layer in range(self.map_layers):
            w = kl.Dense(units=start_filters/2, name='dense_map' + '_' + str(layer))(w)
            w = kl.LeakyReLU(alpha=.2)(w)

        # Synthesis network
        lower_res = None
        for resolution in range(conv_loop):
            filters = max(start_filters // 2**(resolution+1), 4)
            if resolution == 0:  # 4x4
                with tf.variable_scope('Constant'):
                    x = kl.Dense(units=4 * 4 * filters)(constant)
                    x = kl.Reshape([4, 4, filters])(x)
                    x = kl.Lambda(lambda x: add_noise(x, self.batch_size))(x)
                    x = kl.LeakyReLU(alpha=.2)(x)

                    # Adaptive instance normalization and style moderation.
                    x = kl.Lambda(lambda x: instance_norm(x))(x)
                    style_mul = kl.Dense(units=filters, name='dense_' + str(resolution) + '_0_0')(w)
                    style_add = kl.Dense(units=filters, bias_initializer='Ones',
                                         name='dense_' + str(resolution) + '_0_1')(w)
                    x = kl.Multiply()([x, style_mul])
                    x = kl.Add()([x, style_add])

                # Single convolution.
                with tf.variable_scope('Conv_' + str(resolution) + '_0'):
                    x = kl.Conv2D(filters=filters, kernel_size=3, padding='same',
                                  name='conv_' + str(resolution) + '_0')(x)
                    x = kl.Lambda(lambda x: add_noise(x, self.batch_size))(x)
                    x = kl.LeakyReLU(alpha=.2)(x)

                    # Adaptive instance normalization operation and style moderation.
                    x = kl.Lambda(lambda x: instance_norm(x))(x)
                    style_mul = kl.Dense(units=filters, name='dense_' + str(resolution) + '_1_0')(w)
                    style_add = kl.Dense(units=filters, bias_initializer='Ones',
                                         name='dense_' + str(resolution) + '_1_1')(w)
                    x = kl.Multiply()([x, style_mul])
                    x = kl.Add()([x, style_add])
            else:
                x = kl.UpSampling2D()(x)
                for conv in range(2):
                    with tf.variable_scope('Conv_' + str(resolution) + '_' + str(conv)):
                        x = kl.Conv2D(filters=filters, kernel_size=3, padding='same',
                                      name='conv_' + str(resolution) + '_' + str(conv))(x)
                        x = kl.Lambda(lambda x: add_noise(x, self.batch_size))(x)
                        x = kl.LeakyReLU(alpha=.2)(x)

                        # Adaptive instance normalization operation and style moderation.
                        x = kl.Lambda(lambda x: instance_norm(x))(x)
                        style_mul = kl.Dense(units=filters, name='dense_' + str(resolution) + '_' + str(conv) + '_0')(w)
                        style_add = kl.Dense(units=filters, bias_initializer='Ones',
                                             name='dense_' + str(resolution) + '_' + str(conv) + '_1')(w)
                        x = kl.Multiply()([x, style_mul])
                        x = kl.Add()([x, style_add])
            if resolution == conv_loop - 1 and conv_loop > 1:
                lower_res = x

        # Conversion to 3-channel color
        # This is explicitly defined so we can reuse it for the upsampled lower-resolution frames as well
        convert_to_image = kl.Conv2D(filters=3, kernel_size=1, strides=1, padding='same',
                                     kernel_initializer=self.conv_init, use_bias=True, activation='tanh',
                                     name='conv_to_img_'+str(x.get_shape().as_list()[-1]))
        x = convert_to_image(x)

        # Fade output of previous resolution stage into final resolution stage
        if self.fade and lower_res:
            lower_upsampled = kl.UpSampling2D()(lower_res)
            lower_upsampled = convert_to_image(lower_upsampled)
            x = kl.Lambda(lambda x, y, alpha: blend_resolutions(x, y, alpha))([x, lower_upsampled, fade])

        return tf.keras.models.Model(inputs=[z, conditioning, fade, constant], outputs=x, name='generator')

    def discriminator_model(self, out_size, max_filters=512):

        # Fading function
        def blend_resolutions(upper, lower, alpha):
            upper = tf.multiply(upper, alpha)
            lower = tf.multiply(lower, tf.subtract(1.0, alpha)[..., tf.newaxis, tf.newaxis, tf.newaxis])
            return kl.Add()([upper, lower])

        conv_loop = int(np.log2(out_size)) - 3

        img = kl.Input(shape=(out_size, out_size, 3,))
        conditioning = kl.Input(shape=(self.num_classes,))
        fade = kl.Input(shape=(1,))

        # Convert from RGB
        start_filters = int(max(max_filters/2**conv_loop, 4))
        converted = kl.Conv2D(filters=int(start_filters/2), kernel_size=1, strides=1, padding='same',
                      kernel_initializer=self.conv_init, use_bias=True, name='conv_from_img_'+str(out_size))(img)
        # First convolution downsamples by factor of 2
        x = kl.Conv2D(filters=start_filters, kernel_size=4, strides=2, padding='same',
                      kernel_initializer=self.conv_init,
                      name='conv_'+str(out_size/2)+'_'+str(start_filters))(converted)

        # Calculate discriminator score using alpha-blended combination of new discriminator layer outputs
        # versus downsampled version of input videos
        if self.fade:
            downsampled = kl.AveragePooling2D(pool_size=(2, 2), padding='same')(converted)
            x = kl.Lambda(lambda args: blend_resolutions(args[0], args[1], args[2]))([x, downsampled, fade])
        x = kl.Lambda(lambda x: tf.contrib.layers.layer_norm(x))(x)
        x = kl.LeakyReLU(.2)(x)

        for resolution in range(conv_loop):
            filters = start_filters * 2**(resolution + 1)
            x = kl.Conv2D(filters=filters, kernel_size=4, strides=2, padding='same',
                          kernel_initializer=self.conv_init,
                          name='conv_' + str(out_size / 2**(resolution+2))+'_'+str(filters))(x)
            x = kl.Lambda(lambda x: tf.contrib.layers.layer_norm(x))(x)
            x = kl.LeakyReLU(.2)(x)

        # Convert to single value
        x = kl.Conv2D(filters=1, kernel_size=4, strides=2, padding='same',
                      kernel_initializer=self.conv_init, name='conv_1')(x)
        x = kl.LeakyReLU(.2)(x)
        x = kl.Flatten()(x)
        x = tf.concat((x, conditioning), 1)
        x = kl.Dense(1, kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01),
                     name='dense_'+str(x.get_shape().as_list()[-1]))(x)

        return tf.keras.models.Model(inputs=[img, conditioning, fade], outputs=x, name='discriminator')

    def update_models(self, size):
        # Updates generator and discriminator models to add new layers corresponding to next resolution size
        # Retains weights previously learned from lower resolutions
        new_size = size*2
        new_generator, new_discriminator = self.generator_model(new_size), self.discriminator_model(new_size)
        new_gen_layers = [layer.name for layer in new_generator.layers]
        new_disc_layers = [layer.name for layer in new_discriminator.layers]
        for layer in self.generator.layers:
            if ('dense' in layer.name or 'conv' in layer.name) and layer.name in new_gen_layers:
                print('Updating ', layer.name)
                new_generator.get_layer(layer.name).set_weights(self.generator.get_layer(layer.name).get_weights())
        for layer in self.discriminator.layers:
            if ('dense' in layer.name or 'conv' in layer.name) and layer.name in new_disc_layers:
                print('Updating ', layer.name)
                new_discriminator.get_layer(layer.name).set_weights(self.discriminator.get_layer(layer.name).get_weights())
        self.generator, self.discriminator = new_generator, new_discriminator
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.gen_optimizer,
                                              discriminator_optimizer=self.disc_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
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

    def discriminator_loss(self, real_imgs, generated_imgs, real_disc, generated_disc, conditioning, fade, gp_lambda=10):
        # WGAN-GP loss: https://arxiv.org/pdf/1704.00028.pdf
        # Difference between critic scores received by generated output vs real video
        # Lower values mean that the real video samples are receiving higher scores, therefore
        # gradient descent maximizes discriminator accuracy
        out_size = real_imgs.get_shape().as_list()
        d_cost = tf.reduce_mean(generated_disc) - tf.reduce_mean(real_disc)
        alpha = tf.random_uniform(
            shape=[self.batch_size, 1],
            minval=0.,
            maxval=1.
        )
        dim = out_size[2]**2 * 3
        real = tf.reshape(real_imgs, [self.batch_size, dim])
        fake = tf.reshape(generated_imgs, [self.batch_size, dim])
        diff = fake - real
        # Real imgs adjusted by randomly weighted difference between real vs generated
        interpolates = real + (alpha * diff)
        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            interpolates_reshaped = tf.reshape(interpolates, out_size)
            interpolates_disc = self.discriminator([interpolates_reshaped, conditioning, fade])
        # Gradient of critic score wrt interpolated imgs
        gradients = tape.gradient(interpolates_disc, [interpolates])[0]
        # Euclidean norm of gradient for each sample
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        # Gradient norm penalty is the average distance from 1
        gradient_penalty = tf.reduce_mean((norm - 1.) ** 2)

        return d_cost + gp_lambda * gradient_penalty

    def generate(self, epoch, save_dir, num_out, conditioning):
        gen_noise = tf.random_normal([num_out, self.z_dim])
        fade, ones = tf.constant(1, shape=[num_out, 1]), tf.ones((self.batch_size, 1))
        generated_imgs = self.generator(inputs=[gen_noise, conditioning, fade, ones], training=False)
        self.save_img(generated_imgs, save_dir, name=str(generated_imgs[0].shape[-2]) + '_' + str(epoch) + '_')

    def save_img(self, img_tensor, save_dir, name):
        img = tf.cast(255 * (img_tensor + 1)/2, tf.uint8)
        for i, ind_img in enumerate(img):
            encoded = tf.image.encode_jpeg(ind_img)
            tf.write_file(os.path.join(save_dir, name + str(i) + '.jpg'), encoded)
