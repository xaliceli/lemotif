"""
models.py
Conditional generator models
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl

from utils import instance_norm


# Fading function
def blend_resolutions(upper, lower, alpha):
    upper = tf.multiply(upper, alpha[..., tf.newaxis, tf.newaxis])
    lower = tf.multiply(lower, tf.subtract(1.0, alpha)[..., tf.newaxis, tf.newaxis])
    return kl.Add()([upper, lower])


def cond_stylegan(out_size, fade_outputs, z_dim, num_classes, conv_init, batch_size,
                  map_layers=4, start_size=8, start_filters=512, n_channels=3):
    # For now we start at 8x8 and upsample by 2x each time
    conv_loop = int(np.log2(out_size / start_size)) + 2

    z = kl.Input(shape=(z_dim,))
    fade = kl.Input(shape=(1,))
    conditioning = kl.Input(shape=(num_classes,))
    constant = kl.Input(shape=(1,))

    # Mapping network
    # w = tf.concat((z, conditioning), 1)  # Concatenate noise input with condition labels
    w = kl.Concatenate(axis=-1)([z, conditioning])

    for layer in range(map_layers):
        w = kl.Dense(units=start_filters, name='dense_map' + '_' + str(layer))(w)
        w = kl.LeakyReLU(alpha=.2)(w)

    # Synthesis network
    lower_res = None
    for resolution in range(conv_loop):
        filters = max(start_filters // 2 ** (resolution), 4)
        if resolution == 0:  # 4x4
            with tf.variable_scope('Constant'):
                x = kl.Dense(units=4 * 4 * filters)(constant)
                x = kl.Reshape([4, 4, filters])(x)
                # x = kl.Lambda(lambda x: add_noise(x, self.batch_size))(x)
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
                # x = kl.Lambda(lambda x: add_noise(x, self.batch_size))(x)
                x = kl.LeakyReLU(alpha=.2)(x)

                # Adaptive instance normalization operation and style moderation.
                x = kl.Lambda(lambda x: instance_norm(x))(x)
                style_mul = kl.Dense(units=filters, name='dense_' + str(resolution) + '_1_0')(w)
                style_add = kl.Dense(units=filters, bias_initializer='Ones',
                                     name='dense_' + str(resolution) + '_1_1')(w)
                x = kl.Multiply()([x, style_mul])
                x = kl.Add()([x, style_add])
        else:
            x = kl.UpSampling2D(interpolation='bilinear')(x)
            for conv in range(2):
                with tf.variable_scope('Conv_' + str(resolution) + '_' + str(conv)):
                    x = kl.Conv2D(filters=filters, kernel_size=3, padding='same',
                                  name='conv_' + str(resolution) + '_' + str(conv))(x)
                    # x = kl.Lambda(lambda x: add_noise(x, self.batch_size))(x)
                    x = kl.LeakyReLU(alpha=.2)(x)

                    # Adaptive instance normalization operation and style moderation.
                    x = kl.Lambda(lambda x: instance_norm(x))(x)
                    style_mul = kl.Dense(units=filters, name='dense_' + str(resolution) + '_' + str(conv) + '_0')(w)
                    style_add = kl.Dense(units=filters, bias_initializer='Ones',
                                         name='dense_' + str(resolution) + '_' + str(conv) + '_1')(w)
                    x = kl.Multiply()([x, style_mul])
                    x = kl.Add()([x, style_add])
        if resolution == conv_loop - 2 and conv_loop > 1:
            lower_res = x

    # Conversion to 3-channel color
    convert_to_image = kl.Conv2D(filters=n_channels, kernel_size=1, strides=1, padding='same',
                                 kernel_initializer=conv_init, use_bias=True, activation='tanh',
                                 name='conv_to_img_' + str(x.get_shape().as_list()[-1]))
    pre_img_filters = x.get_shape().as_list()[-1]
    x = convert_to_image(x)

    if fade_outputs:
        if lower_res.get_shape().as_list()[-1] == pre_img_filters:
            convert_to_image_lower = convert_to_image
        else:
            convert_to_image_lower = kl.Conv2D(filters=3, kernel_size=1, strides=1, padding='same',
                                               kernel_initializer=conv_init, use_bias=True, activation='tanh',
                                               name='conv_to_img_' + str(lower_res.get_shape().as_list()[-1]))
        # Fade output of previous resolution stage into final resolution stage
        lower_upsampled = kl.UpSampling2D(interpolation='bilinear')(lower_res)
        lower_upsampled = convert_to_image_lower(lower_upsampled)
        x = kl.Lambda(lambda args: blend_resolutions(args[0], args[1], args[2]))([x, lower_upsampled, fade])

    return tf.keras.models.Model(inputs=[z, conditioning, fade, constant], outputs=x, name='generator')


def cond_progan(out_size, fade_outputs, z_dim, num_classes, conv_init, batch_size, cond_start=False,
                start_size=8, start_filters=512, n_channels=3):
    # For now we start at 8x8 and upsample by 2x each time
    conv_loop = int(np.log2(out_size / start_size)) + 2

    z = kl.Input(shape=(z_dim,))
    fade = kl.Input(shape=(1,))
    conditioning = kl.Input(shape=(num_classes,))
    if cond_start:
        # z_cond = tf.concat((z, conditioning), 1)
        z_cond = kl.Concatenate(axis=-1)([z, conditioning])
    else:
        z_cond = z

    # Synthesis network
    lower_res = None
    for resolution in range(conv_loop):
        filters = max(start_filters // 2 ** (resolution), 4)
        if resolution == 0:  # 4x4
            with tf.variable_scope('Dense_' + str(resolution) + '_0'):
                x = kl.Dense(units=4 * 4 * filters, name='dense_1')(z_cond)
                x = kl.Reshape([4, 4, filters])(x)
                x = kl.BatchNormalization(name='bn_'+str(resolution) + '_0_' +str(filters))(x)
                x = kl.LeakyReLU(alpha=.2)(x)

            # Single convolution.
            with tf.variable_scope('Conv_' + str(resolution) + '_1'):
                x = kl.Conv2D(filters=filters, kernel_size=3, padding='same', kernel_initializer=conv_init,
                              name='conv_' + str(resolution) + '_0')(x)
                x = kl.BatchNormalization(name='bn_'+str(resolution) + '_1_' +str(filters))(x)
                x = kl.LeakyReLU(alpha=.2)(x)
        else:
            x = kl.UpSampling2D(interpolation='bilinear')(x)
            for conv in range(2):
                with tf.variable_scope('Conv_' + str(resolution) + '_' + str(conv)):
                    x = kl.Conv2D(filters=filters, kernel_size=3, padding='same', kernel_initializer=conv_init,
                                  name='conv_' + str(resolution) + '_' + str(conv))(x)
                    x = kl.BatchNormalization(name='bn_'+str(resolution) + '_' + str(conv) + '_' +str(filters))(x)
                    x = kl.LeakyReLU(alpha=.2)(x)
        if resolution == conv_loop - 2 and conv_loop > 1:
            lower_res = x

    if not cond_start:
        cond_tiled = kl.Lambda(tf.tile, arguments={'multiples': [1, out_size ** 2]})(conditioning)
        cond_layers = kl.Reshape((out_size, out_size, num_classes))(cond_tiled)
        x = kl.Concatenate(axis=-1)([x, cond_layers])
        x = kl.Conv2D(filters=filters, kernel_size=1, strides=1, padding='same', kernel_initializer=conv_init,
                      use_bias=True, name='conv_pre_img_' + str(out_size) + '_' + str(x.get_shape().as_list()[-1]))(x)

    # Conversion to 3-channel color
    convert_to_image = kl.Conv2D(filters=n_channels, kernel_size=1, strides=1, padding='same',
                                 kernel_initializer=conv_init, use_bias=True, activation='tanh',
                                 name='conv_to_img_' + str(x.get_shape().as_list()[-1]))
    pre_img_filters = x.get_shape().as_list()[-1]
    x = convert_to_image(x)

    if fade_outputs:
        if lower_res.get_shape().as_list()[-1] == pre_img_filters:
            convert_to_image_lower = convert_to_image
        else:
            convert_to_image_lower = kl.Conv2D(filters=3, kernel_size=1, strides=1, padding='same',
                                               kernel_initializer=conv_init, use_bias=True, activation='tanh',
                                               name='conv_to_img_' + str(lower_res.get_shape().as_list()[-1]))
        # Fade output of previous resolution stage into final resolution stage
        if not cond_start:
            lower_cond_tiled = kl.Lambda(tf.tile, arguments={'multiples': [1, int((out_size/2) ** 2)]})(conditioning)
            lower_cond_layers = kl.Reshape((int(out_size/2), int(out_size/2), num_classes))(lower_cond_tiled)
            lower_res = kl.Concatenate(axis=-1)([lower_res, lower_cond_layers])
            ls = lower_res.get_shape().as_list()
            lower_res = kl.Conv2D(filters=filters*2, kernel_size=1, strides=1, padding='same',
                                  kernel_initializer=conv_init, use_bias=True,
                                  name='conv_pre_img_' + str(ls[1]) + '_' + str(ls[-1]))(lower_res)
        lower_upsampled = kl.UpSampling2D(interpolation='bilinear')(lower_res)
        lower_upsampled = convert_to_image_lower(lower_upsampled)
        x = kl.Lambda(lambda args: blend_resolutions(args[0], args[1], args[2]))([x, lower_upsampled, fade])

    return tf.keras.models.Model(inputs=[z, conditioning, fade], outputs=x, name='generator')


def encoder_basic(in_size, z_dim, conv_init, max_filters=512):

    conv_loop = int(np.log2(in_size)) - 2

    img = kl.Input(shape=(in_size, in_size, 3,))

    start_filters = int(max(max_filters / 2 ** conv_loop, 4))
    x = img
    for resolution in range(conv_loop):
        filters = start_filters * 2 ** (resolution + 1)
        x = kl.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
                      kernel_initializer=conv_init,
                      name='conv_' + str(in_size / 2 ** (resolution)) + '_0_' + '_' + str(filters))(x)
        x = kl.BatchNormalization()(x)
        x = kl.LeakyReLU(.2)(x)

        x = kl.Conv2D(filters=filters, kernel_size=3, strides=2, padding='same',
                      kernel_initializer=conv_init,
                      name='conv_' + str(in_size / 2 ** (resolution)) + '_1_' + '_' + str(filters))(x)
        x = kl.BatchNormalization()(x)
        x = kl.LeakyReLU(.2)(x)

    # Convert to z
    x = kl.Flatten()(x)
    z = kl.Dense(z_dim, name='dense_to_z_' + str(z_dim) + '_' + str(x.get_shape().as_list()[-1]))(x)

    return tf.keras.models.Model(inputs=[img], outputs=[z], name='encoder')


def encoder_residual(in_size, z_dim, conv_init, max_filters=512):
    img = kl.Input(shape=(in_size, in_size, 3,))

    conv_loop = int(np.log2(in_size)) - 1
    start_filters = int(max(max_filters / 2 ** (conv_loop - 1), 4))
    x = kl.Conv2D(filters=start_filters, kernel_size=1, strides=1, padding='same', kernel_initializer=conv_init,
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                  name='conv_from_img_' + str(start_filters))(img)

    for resolution in range(conv_loop):
        filters = start_filters * 2 ** (resolution)
        strides = 2 if resolution > 0 else 1
        x1 = kl.Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same', kernel_initializer=conv_init,
                       kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                       name='conv_' + str(in_size / 2 ** (resolution)) + '_0_' + '_' + str(filters))(x)
        x1 = kl.BatchNormalization()(x1)
        x1 = kl.LeakyReLU(.2)(x1)

        x2 = kl.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', kernel_initializer=conv_init,
                       kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                       name='conv_' + str(in_size / 2 ** (resolution)) + '_1_' + '_' + str(filters))(x1)
        x2 = kl.BatchNormalization()(x2)

        if resolution > 0:
            x = kl.Conv2D(filters=filters, kernel_size=1, strides=2, padding='same', kernel_initializer=conv_init,
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                          name='conv_resid_down_' + str(x.get_shape().as_list()[-1]))(x)
        x = kl.Add()([x, x2])
        x = kl.LeakyReLU(.2)(x)

    # Convert to z
    x = kl.Flatten()(x)
    z = kl.Dense(z_dim, name='dense_to_z_' + str(z_dim) + '_' + str(x.get_shape().as_list()[-1]))(x)

    return tf.keras.models.Model(inputs=[img], outputs=[z], name='encoder')


def decoder_basic(out_size, z_dim, conv_init, max_filters=512):

    conv_loop = int(np.log2(out_size / 8)) + 2

    z = kl.Input(shape=(z_dim,))

    for resolution in range(conv_loop):
        filters = max(max_filters // 2 ** (resolution), 4)
        if resolution == 0:  # 4x4
            with tf.variable_scope('Dense_' + str(resolution) + '_0'):
                x = kl.Dense(units=4 * 4 * filters, name='dense_' + str(filters))(z)
                x = kl.Reshape([4, 4, filters])(x)
                x = kl.BatchNormalization()(x)
                x = kl.LeakyReLU(alpha=.2)(x)

            # Single convolution.
            with tf.variable_scope('Conv_' + str(resolution) + '_1'):
                x = kl.Conv2D(filters=filters, kernel_size=3, padding='same', kernel_initializer=conv_init,
                              name='conv_' + str(resolution) + '_0')(x)
                x = kl.BatchNormalization()(x)
                x = kl.LeakyReLU(alpha=.2)(x)
        else:
            x = kl.UpSampling2D(interpolation='bilinear')(x)
            for conv in range(2):
                with tf.variable_scope('Conv_' + str(resolution) + '_' + str(conv)):
                    x = kl.Conv2D(filters=filters, kernel_size=3, padding='same', kernel_initializer=conv_init,
                                  name='conv_' + str(resolution) + '_' + str(conv))(x)
                    x = kl.BatchNormalization()(x)
                    x = kl.LeakyReLU(alpha=.2)(x)

    # Conversion to 3-channel color
    x = kl.Conv2D(filters=3, kernel_size=1, strides=1, padding='same',
                  kernel_initializer=conv_init, use_bias=True, activation='tanh',
                  name='conv_to_img_' + str(x.get_shape().as_list()[-1]))(x)

    return tf.keras.models.Model(inputs=[z], outputs=[x], name='decoder')


def discriminator_basic(out_size, conv_init, max_filters=512):

    conv_loop = int(np.log2(out_size)) - 2

    img = kl.Input(shape=(out_size, out_size, 3,))

    start_filters = int(max(max_filters / 2 ** (conv_loop - 1), 4))
    x = img
    for resolution in range(conv_loop):
        filters = start_filters * 2 ** resolution
        x = kl.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', kernel_initializer=conv_init,
                      name='conv_' + str(out_size / 2 ** (resolution)) + '_0_' + '_' + str(filters))(x)
        x = kl.BatchNormalization()(x)
        x = kl.LeakyReLU(.2)(x)

        x = kl.Conv2D(filters=filters, kernel_size=3, strides=2, padding='same', kernel_initializer=conv_init,
                      name='conv_' + str(out_size / 2 ** (resolution)) + '_1_' + '_' + str(filters))(x)
        x = kl.BatchNormalization()(x)
        x = kl.LeakyReLU(.2)(x)

    # Convert to single value
    x = kl.Flatten()(x)
    disc_out = kl.Dense(1, name='dense_to_score_' + str(x.get_shape().as_list()[-1]))(x)

    return tf.keras.models.Model(inputs=[img], outputs=[disc_out], name='discriminator')


def discriminator_residual(out_size, conv_init, max_filters=512):
    img = kl.Input(shape=(out_size, out_size, 3,))

    conv_loop = int(np.log2(out_size)) - 1
    start_filters = int(max(max_filters / 2 ** (conv_loop - 1), 4))
    x = kl.Conv2D(filters=start_filters, kernel_size=1, strides=1, padding='same', kernel_initializer=conv_init,
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                  name='conv_from_img_' + str(start_filters))(img)

    for resolution in range(conv_loop):
        filters = start_filters * 2 ** (resolution)
        strides = 2 if resolution > 0 else 1
        x1 = kl.Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same', kernel_initializer=conv_init,
                       kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                       name='conv_' + str(out_size / 2 ** (resolution)) + '_0_' + '_' + str(filters))(x)
        x1 = kl.BatchNormalization()(x1)
        x1 = kl.LeakyReLU(.2)(x1)

        x2 = kl.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', kernel_initializer=conv_init,
                       kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                       name='conv_' + str(out_size / 2 ** (resolution)) + '_1_' + '_' + str(filters))(x1)
        x2 = kl.BatchNormalization()(x2)

        if resolution > 0:
            x = kl.Conv2D(filters=filters, kernel_size=1, strides=2, padding='same', kernel_initializer=conv_init,
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                          name='conv_resid_down_' + str(x.get_shape().as_list()[-1]))(x)
        x = kl.Add()([x, x2])
        x = kl.LeakyReLU(.2)(x)

    # Convert to single value
    x = kl.Flatten()(x)
    disc_out = kl.Dense(1, name='dense_to_score_' + str(x.get_shape().as_list()[-1]))(x)

    return tf.keras.models.Model(inputs=[img], outputs=[disc_out], name='discriminator')

def discriminator_progressive(out_size, fade_inputs, batch_size, n_channels, num_classes, conv_init, max_filters=512):

    # Fading function
    def blend_resolutions(upper, lower, alpha):
        upper = tf.multiply(upper, alpha[..., tf.newaxis, tf.newaxis])
        lower = tf.multiply(lower, tf.subtract(1.0, alpha)[..., tf.newaxis, tf.newaxis])
        return kl.Add()([upper, lower])

    conv_loop = int(np.log2(out_size)) - 1

    img = kl.Input(shape=(out_size, out_size, n_channels,))
    conditioning = kl.Input(shape=(num_classes,))
    fade = kl.Input(shape=(1,))

    # Convert from RGB
    start_filters = int(max(max_filters / 2 ** (conv_loop), 4))
    x = kl.Conv2D(filters=int(start_filters), kernel_size=1, strides=1, padding='same',
                  kernel_initializer=conv_init, use_bias=True,
                  name='conv_from_img_'+str(out_size))(img)
    x = kl.BatchNormalization(name='bn_from_img_' + str(out_size) + '_' + str(start_filters))(x)
    x = kl.LeakyReLU(.2)(x)

    if fade_inputs:
        # Calculate discriminator score using alpha-blended combination of new discriminator layer outputs
        # versus downsampled version of input images
        downsampled = kl.AveragePooling2D(pool_size=(2, 2), padding='same')(img)
        downsampled = kl.Conv2D(filters=int(start_filters*2), kernel_size=1, strides=1, padding='same',
                                kernel_initializer=conv_init, use_bias=True,
                                name='conv_from_img_'+str(int(out_size/2)))(downsampled)
        downsampled = kl.BatchNormalization(name='bn_from_img_' + str(int(out_size/2)) + '_' + str(int(start_filters*2))
                                            )(downsampled)
        downsampled = kl.LeakyReLU(.2)(downsampled)

    for resolution in range(conv_loop):
        filters = start_filters * 2 ** (resolution + 1)

        # if resolution == conv_loop - 1:
            # x = minibatch_stddev_layer(x, batch_size)
            # x = kl.Lambda(minibatch_stddev_layer, arguments={'batch_size': batch_size})(x)

        x = kl.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
                      kernel_initializer=conv_init,
                      name='conv_' + str(out_size / 2 ** (resolution)) + '_0_' + '_' + str(filters))(x)
        x = kl.BatchNormalization(name='bn_' + str(resolution) + '_0_' + str(filters))(x)
        x = kl.LeakyReLU(.2)(x)

        if resolution < conv_loop - 1:
            x = kl.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=conv_init,
                          name='conv_' + str(out_size / 2 ** (resolution)) + '_1_' + '_' + str(filters))(x)
            x = kl.BatchNormalization(name='bn_' + str(resolution) + '_1_' + str(filters))(x)
            x = kl.LeakyReLU(.2)(x)
            x = kl.AveragePooling2D(pool_size=(2, 2), padding='same')(x)

        if resolution == 0 and fade_inputs:
            x = kl.Lambda(lambda args: blend_resolutions(args[0], args[1], args[2]))([x, downsampled, fade])

    # Convert to single value
    x = kl.Flatten()(x)
    x = kl.Dense(max_filters, name='dense_pre_'+str(x.get_shape().as_list()[-1]))(x)
    x = kl.LeakyReLU(.2)(x)

    # Label prediction
    labels_out = kl.Dense(num_classes, name='dense_to_labels_'+str(x.get_shape().as_list()[-1]))(x)

    # Discriminator score
    x = kl.Concatenate(axis=-1)([x, conditioning])

    disc_out = kl.Dense(1, name='dense_to_score_'+str(x.get_shape().as_list()[-1]))(x)

    return tf.keras.models.Model(inputs=[img, conditioning, fade], outputs=[disc_out, labels_out], name='discriminator')
