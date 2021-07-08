import gin
import tensorflow as tf
import tensorflow.python.keras as keras
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Input, Flatten, Lambda, Dense, Reshape, Activation, Conv3D, LeakyReLU, \
    PReLU, Add, Conv3DTranspose, SpatialDropout3D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

from utils import sampling


def ActivationOp(
        layer_in,
        activation_type,
        name=None,
        shared_axes=(1, 2, 3),
        l=0.1,
):
    if (activation_type != 'prelu') & (activation_type != 'leakyrelu'):
        return Activation(activation_type, name=name)(layer_in)
    elif activation_type == 'prelu':
        return PReLU(
            alpha_initializer=keras.initializers.Constant(value=l),
            shared_axes=shared_axes,
            name=name,
        )(layer_in)
    else:
        # TODO: check if alpha should be 0.01 instead
        return LeakyReLU(l)(layer_in)


def ResidualBlock3D(
        layer_in,
        depth=3,
        kernel_size=2,
        filters=None,
        activation='relu',
        kernel_initializer='he_normal',
        name=None,
):
    # creates a residual block with a given depth for 3D input
    # there is NO non-linearity applied to the output! Has to be added manually
    l = Conv3D(
        filters,
        kernel_size,
        padding='same',
        activation='linear',
        kernel_initializer=kernel_initializer,
        name='{}_c0'.format(name),
    )(layer_in)
    for i in range(1, depth):
        a = ActivationOp(
            l,
            activation,
            name='{}_a{}'.format(name, i - 1),
        )
        l = Conv3D(
            filters,
            kernel_size,
            padding='same',
            activation='linear',
            kernel_initializer=kernel_initializer,
            name='{}_c{}'.format(name, i),
        )(a)
    o = Add()([layer_in, l])
    # o = Activation_wrap(o, activation, name='{}_a{}'.format(name,depth))
    return o


def DownConv3D(
        layer_in,
        kernel_size=2,
        strides=(2, 2, 2),
        filters=None,
        activation='relu',
        kernel_initializer='he_normal',
        name=None,
        data_format=None,
):
    if isinstance(strides, int):
        strides = (strides, strides, strides)
    dc = Conv3D(
        filters,
        kernel_size,
        strides=strides,
        padding='valid',
        activation='linear',
        name='{}_dc0'.format(name),
        kernel_initializer=kernel_initializer,
        data_format=data_format,
    )(layer_in)
    dc = ActivationOp(dc, activation, name='{}_a0'.format(name))
    return dc


def UpConv3D(
        layer_in,
        kernel_size=(2, 2, 2),
        strides=None,
        filters=None,
        activation='relu',
        kernel_initializer='he_normal',
        name=None,
        data_format=None,
):
    if strides is None:
        strides = kernel_size
    elif isinstance(strides, int):
        strides = (strides, strides, strides)
    uc = Conv3DTranspose(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        activation='linear',
        name='{}_uc0'.format(name),
        kernel_initializer=kernel_initializer,
        padding='same',
        data_format=data_format,
    )(layer_in)
    uc = ActivationOp(uc, activation, name='{}_a0'.format(name))
    return uc


@gin.configurable(name_or_fn='model', denylist=['input_shape', 'data_format'])
def vae_reg(
        input_shape=(96, 96, 96),
        filters=(8, 16, 32, 64, 128, 256),
        weight_L2=1.,
        weight_reg=.1,
        weight_KL=0.01,
        adam_lr=1e-4,
        adam_decay=.9,
        # TODO remove?
        conv_weight_decay=None,
        dropout_rate=0.,
        dim_latent_space=1024,
        activation='relu',
        rec_activation='linear',
        data_format='channels_last',
):
    input_shape = input_shape + (1,) if data_format == 'channels_last' else (1,) + input_shape

    input = Input(input_shape)
    layer = Conv3D(
        filters=filters[0],
        kernel_size=(3, 3, 3),
        activation=tf.python.keras.layers.LeakyReLU(.1) if activation == 'leakyrelu' else activation,
        padding='SAME',
        kernel_initializer='he_normal',
        name='enc0',
        data_format=data_format,
    )(input)
    layer = DownConv3D(layer, filters=filters[1], name='enc1', data_format=data_format, activation=activation)
    layer = Conv3D(
        filters=filters[2],
        kernel_size=(3, 3, 3),
        activation=tf.python.keras.layers.LeakyReLU(.1) if activation == 'leakyrelu' else activation,
        padding='SAME',
        kernel_initializer='he_normal',
        name='enc3',
        data_format=data_format,
    )(layer)
    layer = DownConv3D(layer, filters=filters[3], name='enc3_5', data_format=data_format, activation=activation)
    layer = DownConv3D(layer, filters=filters[4], name='enc4', data_format=data_format, activation=activation)
    layer = DownConv3D(layer, filters=filters[5], name='enc5', data_format=data_format, activation=activation)
    layer = DownConv3D(layer, filters=filters[5], name='enc6', data_format=data_format, activation=activation)
    layer = SpatialDropout3D(dropout_rate)(layer)
    layer_shape = K.int_shape(layer)
    layer = Flatten()(layer)

    mu, log_var = Dense(dim_latent_space, name='mu')(layer), Dense(dim_latent_space, name='log_var')(layer)
    z = Lambda(sampling, name='z')([mu, log_var])

    layer = Dense(
        layer_shape[1] * layer_shape[2] * layer_shape[3] * layer_shape[4],
        activation=tf.python.keras.layers.LeakyReLU(.1) if activation == 'leakyrelu' else activation,
    )(z)
    layer = Reshape((layer_shape[1], layer_shape[2], layer_shape[3], layer_shape[4]))(layer)

    layer = Conv3D(
        filters=filters[5],
        kernel_size=(3, 3, 3),
        activation=tf.python.keras.layers.LeakyReLU(.1) if activation == 'leakyrelu' else activation,
        padding='SAME',
        kernel_initializer='he_normal',
        name='dc21',
        data_format=data_format,
    )(layer)
    layer = UpConv3D(layer, filters=filters[5], name='dc2', data_format=data_format, activation=activation)
    layer = UpConv3D(layer, filters=filters[4], name='dc3', data_format=data_format, activation=activation)
    layer = UpConv3D(layer, filters=filters[3], name='dc4', data_format=data_format, activation=activation)
    layer = UpConv3D(layer, filters=filters[2], name='dc5', data_format=data_format, activation=activation)
    layer = Conv3D(
        filters=filters[1],
        kernel_size=(3, 3, 3),
        activation=tf.python.keras.layers.LeakyReLU(.1) if activation == 'leakyrelu' else activation,
        padding='same',
        kernel_initializer='he_normal',
        name='dc22',
        data_format=data_format,
    )(layer)
    layer = UpConv3D(layer, filters=filters[0], name='dc6', data_format=data_format, activation=activation)
    reconstruction = Conv3D(filters=1, kernel_size=(3, 3, 3), activation=rec_activation, padding='SAME',
                            kernel_initializer='he_normal', name='reconstruction',
                            data_format=data_format)(layer)

    # regression head
    reg_branch = Dense(
        128,
        activation=tf.python.keras.layers.LeakyReLU(.1) if activation == 'leakyrelu' else activation,
        name='reg_dense_1',
    )(z)
    reg_branch = Dense(
        32,
        activation=tf.python.keras.layers.LeakyReLU(.1) if activation == 'leakyrelu' else activation,
        name='reg_dense_2',
    )(reg_branch)
    reg_branch = Dense(1, name='regression')(reg_branch)

    def kl_loss(*args, **kwargs):
        return tf.reduce_mean(-.5 * (1 + log_var - tf.square(mu) - tf.exp(log_var)))

    def num_active_dims(y_true, y_pred):
        threshold = .1
        _num_active_dims = tf.math.count_nonzero(tf.exp(log_var) < threshold)

        return _num_active_dims

    model = Model([input], [reconstruction, reg_branch, log_var])
    model.compile(
        optimizer=Adam(
            lr=adam_lr,
            decay=adam_decay,
        ),
        loss=['mse', 'mse', kl_loss],
        loss_weights=[weight_L2, weight_reg, weight_KL],
        metrics={log_var.name.split('/')[0]: num_active_dims},
    )

    return model
