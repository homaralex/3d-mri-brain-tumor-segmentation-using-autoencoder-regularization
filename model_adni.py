import gin
import tensorflow.python.keras as keras
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Input, Flatten, Dropout, Dense, Reshape, Activation, Conv3D, LeakyReLU, \
    PReLU, Add, Conv3DTranspose
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam


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


@gin.configurable(name_or_fn='model', denylist=[
    'input_shape',
    'z_score',
    'data_format',
])
def vae_reg(
        input_shape=(96, 96, 96),
        filters=(8, 16, 32, 64, 128, 256),
        weight_L2=0.1,
        weight_KL=0.1,
        adam_lr=1e-4,
        adam_decay=.9,
        # TODO remove?
        conv_weight_decay=None,
        dropout_rate=0.,
        dim_latent_space=1024,
        # whether input is normalized to [0, 1] (we use sigmoid activations for VAE output then)
        z_score=False,
        data_format='channels_last',
):
    assert not z_score, 'z-score not implemented yet'

    input_shape = input_shape + (1,) if data_format == 'channels_last' else (1,) + input_shape

    input = Input(input_shape)
    layer = Conv3D(filters=filters[0], kernel_size=(3, 3, 3), activation='relu', padding='SAME',
                   kernel_initializer='he_normal', name='enc0')(input)
    layer = DownConv3D(layer, filters=filters[1], name='enc1')
    layer = Conv3D(filters=filters[2], kernel_size=(3, 3, 3), activation='relu', padding='SAME',
                   kernel_initializer='he_normal', name='enc3')(layer)
    layer = DownConv3D(layer, filters=filters[3], name='enc3_5')
    layer = DownConv3D(layer, filters=filters[4], name='enc4')
    layer = DownConv3D(layer, filters=filters[5], name='enc5')
    layer = DownConv3D(layer, filters=filters[5], name='enc6')
    layer_shape = K.int_shape(layer)
    layer = Flatten()(layer)
    # TODO spatial dropout?
    layer = Dropout(dropout_rate)(layer)

    layer = Dense(dim_latent_space, activation='relu')(layer)

    layer = Dense(layer_shape[1] * layer_shape[2] * layer_shape[3] * layer_shape[4], activation='relu')(layer)
    layer = Reshape((layer_shape[1], layer_shape[2], layer_shape[3], layer_shape[4]))(layer)

    layer = Conv3D(filters=filters[5], kernel_size=(3, 3, 3), activation='relu', padding='SAME',
                   kernel_initializer='he_normal', name='dc21')(layer)
    layer = UpConv3D(layer, filters=filters[5], name='dc2')
    layer = UpConv3D(layer, filters=filters[4], name='dc3')
    layer = UpConv3D(layer, filters=filters[3], name='dc4')
    layer = UpConv3D(layer, filters=filters[2], name='dc5')
    layer = Conv3D(filters=filters[1], kernel_size=(3, 3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal', name='dc22')(layer)
    layer = UpConv3D(layer, filters=filters[0], name='dc6')
    layer = Conv3D(filters=1, kernel_size=(3, 3, 3), activation='linear', padding='SAME',
                   kernel_initializer='he_normal', name='cd7')(layer)

    model = Model([input], [layer])
    model.compile(
        optimizer=Adam(
            lr=adam_lr,
            decay=adam_decay,
        ),
        loss='mean_squared_error',
        metrics=['mse'],
    )

    return model
