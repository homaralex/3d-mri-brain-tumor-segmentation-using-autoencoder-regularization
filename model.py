# Keras implementation of the paper:
# 3D MRI Brain Tumor Segmentation Using Autoencoder Regularization
# by Myronenko A. (https://arxiv.org/pdf/1810.11654.pdf)
# Author of this code: Suyog Jadhav (https://github.com/IAmSUyogJadhav)
import gin
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv3D, Activation, Add, UpSampling3D, Lambda, Dense
from tensorflow.keras.layers import Input, Reshape, Flatten, SpatialDropout3D
from tensorflow.keras.optimizers import Adam as adam
from tensorflow.keras.models import Model
from tensorflow_addons.layers import GroupNormalization


@gin.configurable(allowlist=['conv_3d_cls'])
def green_block(
        inp,
        filters,
        conv_3d_cls=gin.REQUIRED,
        data_format='channels_last',
        name=None,
):
    """
    green_block(inp, filters, name=None)
    ------------------------------------
    Implementation of the special residual block used in the paper. The block
    consists of two (GroupNorm --> ReLu --> 3x3x3 non-strided Convolution)
    units, with a residual connection from the input `inp` to the output. Used
    internally in the model. Can be used independently as well.

    Parameters
    ----------
    `inp`: An keras.layers.layer instance, required
        The keras layer just preceding the green block.
    `filters`: integer, required
        No. of filters to use in the 3D convolutional block. The output
        layer of this green block will have this many no. of channels.
    `data_format`: string, optional
        The format of the input data. Must be either 'chanels_first' or
        'channels_last'. Defaults to `channels_first`, as used in the paper.
    `name`: string, optional
        The name to be given to this green block. Defaults to None, in which
        case, keras uses generated names for the involved layers. If a string
        is provided, the names of individual layers are generated by attaching
        a relevant prefix from [GroupNorm_, Res_, Conv3D_, Relu_, ], followed
        by _1 or _2.

    Returns
    -------
    `out`: A keras.layers.Layer instance
        The output of the green block. Has no. of channels equal to `filters`.
        The size of the rest of the dimensions remains same as in `inp`.
    """
    inp_res = conv_3d_cls(
        filters=filters,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format=data_format,
        name=f'Res_{name}' if name else None)(inp)

    # axis=1 for channels_first data format
    # No. of groups = 8, as given in the paper
    x = GroupNormalization(
        groups=8,
        axis=1 if data_format == 'channels_first' else -1,
        name=f'GroupNorm_1_{name}' if name else None)(inp)
    x = Activation('relu', name=f'Relu_1_{name}' if name else None)(x)
    x = conv_3d_cls(
        filters=filters,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        data_format=data_format,
        name=f'Conv3D_1_{name}' if name else None)(x)

    x = GroupNormalization(
        groups=8,
        axis=1 if data_format == 'channels_first' else -1,
        name=f'GroupNorm_2_{name}' if name else None)(x)
    x = Activation('relu', name=f'Relu_2_{name}' if name else None)(x)
    x = conv_3d_cls(
        filters=filters,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        data_format=data_format,
        name=f'Conv3D_2_{name}' if name else None)(x)

    out = Add(name=f'Out_{name}' if name else None)([x, inp_res])
    return out


# From keras-team/keras/blob/master/examples/variational_autoencoder.py
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = tf.random.normal(shape=(batch, dim))

    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def loss_gt(e=1e-8, data_format='channels_last'):
    """
    loss_gt(e=1e-8)
    ------------------------------------------------------
    Since keras does not allow custom loss functions to have arguments
    other than the true and predicted labels, this function acts as a wrapper
    that allows us to implement the custom loss used in the paper. This function
    only calculates - L<dice> term of the following equation. (i.e. GT Decoder part loss)
    
    L = - L<dice> + weight_L2 ∗ L<L2> + weight_KL ∗ L<KL>
    
    Parameters
    ----------
    `e`: Float, optional
        A small epsilon term to add in the denominator to avoid dividing by
        zero and possible gradient explosion.
    Returns
    -------
    loss_gt_(y_true, y_pred): A custom keras loss function
        This function takes as input the predicted and ground labels, uses them
        to calculate the dice loss.
        
    """

    def loss_gt_(y_true, y_pred):
        axes = [-3, -2, -1] if data_format == 'channels_first' else [-4, -3, -2]

        intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=axes) + e / 2
        dn = tf.reduce_sum(tf.square(y_true), axis=axes) + tf.reduce_sum(tf.square(y_pred), axis=axes) + e

        return - tf.reduce_mean(2 * intersection / dn)

    return loss_gt_


@gin.configurable(name_or_fn='model', denylist=[
    'input_shape',
    'output_channels',
    'z_score',
    'data_format',
    'dice_e',
])
def build_model(
        input_shape=(160, 192, 128, 4),
        output_channels=3,
        weight_dice=1,
        weight_L2=0.1,
        weight_KL=0.1,
        adam_lr=1e-4,
        adam_decay=.9,
        dice_e=1e-8,
        conv_weight_decay=None,
        dropout_rate=.2,
        dim_latent_space=128,
        # whether input is normalized to [0, 1] (we use sigmoid activations for VAE output then)
        z_score=False,
        data_format='channels_last',
        shared_latent_space=False,
):
    """
    build_model(input_shape=(4, 160, 192, 128), output_channels=3, weight_L2=0.1, weight_KL=0.1)
    -------------------------------------------
    Creates the model used in the BRATS2018 winning solution
    by Myronenko A. (https://arxiv.org/pdf/1810.11654.pdf)

    Parameters
    ----------
    `input_shape`: A 4-tuple, optional.
        Shape of the input image. Must be a 4D image of shape (c, H, W, D),
        where, each of H, W and D are divisible by 2^4, and c is divisible by 4.
        Defaults to the crop size used in the paper, i.e., (4, 160, 192, 128).
    `output_channels`: An integer, optional.
        The no. of channels in the output. Defaults to 3 (BraTS 2018 format).
    `weight_L2`: A real number, optional
        The weight to be given to the L2 loss term in the loss function. Adjust to get best
        results for your task. Defaults to 0.1.
    `weight_KL`: A real number, optional
        The weight to be given to the KL loss term in the loss function. Adjust to get best
        results for your task. Defaults to 0.1.
    `dice_e`: Float, optional
        A small epsilon term to add in the denominator of dice loss to avoid dividing by
        zero and possible gradient explosion. This argument will be passed to loss_gt function.


    Returns
    -------
    `model`: A keras.models.Model instance
        The created model.
    """
    if data_format == 'channels_first':
        c, H, W, D = input_shape
    else:
        H, W, D, c = input_shape
    assert len(input_shape) == 4, "Input shape must be a 4-tuple"
    # assert (c % 4) == 0, "The no. of channels must be divisible by 4"
    assert (H % 16) == 0 and (W % 16) == 0 and (D % 16) == 0, \
        "All the input dimensions must be divisible by 16"

    conv_3d_cls = Conv3D
    if conv_weight_decay is not None:
        conv_3d_cls = gin.external_configurable(Conv3D, 'Conv3D')
        gin.bind_parameter('Conv3D.kernel_regularizer', keras.regularizers.L1L2(l2=conv_weight_decay))
    gin.bind_parameter('green_block.conv_3d_cls', conv_3d_cls)

    # -------------------------------------------------------------------------
    # Encoder
    # -------------------------------------------------------------------------

    ## Input Layer
    inp = Input(input_shape)

    ## The Initial Block
    x = conv_3d_cls(
        filters=32,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        data_format=data_format,
        name='Input_x1')(inp)

    ## Dropout (0.2)
    x = SpatialDropout3D(dropout_rate, data_format=data_format)(x)

    ## Green Block x1 (output filters = 32)
    x1 = green_block(x, 32, name='x1', data_format=data_format)
    x = conv_3d_cls(
        filters=32,
        kernel_size=(3, 3, 3),
        strides=2,
        padding='same',
        data_format=data_format,
        name='Enc_DownSample_32')(x1)

    ## Green Block x2 (output filters = 64)
    x = green_block(x, 64, name='Enc_64_1', data_format=data_format)
    x2 = green_block(x, 64, name='x2', data_format=data_format)
    x = conv_3d_cls(
        filters=64,
        kernel_size=(3, 3, 3),
        strides=2,
        padding='same',
        data_format=data_format,
        name='Enc_DownSample_64')(x2)

    ## Green Blocks x2 (output filters = 128)
    x = green_block(x, 128, name='Enc_128_1', data_format=data_format)
    x3 = green_block(x, 128, name='x3', data_format=data_format)
    x = conv_3d_cls(
        filters=128,
        kernel_size=(3, 3, 3),
        strides=2,
        padding='same',
        data_format=data_format,
        name='Enc_DownSample_128')(x3)

    ## Green Blocks x4 (output filters = 256)
    x = green_block(x, 256, name='Enc_256_1', data_format=data_format)
    x = green_block(x, 256, name='Enc_256_2', data_format=data_format)
    x = green_block(x, 256, name='Enc_256_3', data_format=data_format)
    x4 = green_block(x, 256, name='x4', data_format=data_format)

    # -------------------------------------------------------------------------
    # Decoder
    # -------------------------------------------------------------------------

    ## GT (Groud Truth) Part
    # -------------------------------------------------------------------------

    if not shared_latent_space:
        ### Green Block x1 (output filters=128)
        x = conv_3d_cls(
            filters=128,
            kernel_size=(1, 1, 1),
            strides=1,
            data_format=data_format,
            name='Dec_GT_ReduceDepth_128')(x4)
        x = UpSampling3D(
            size=2,
            data_format=data_format,
            name='Dec_GT_UpSample_128')(x)
        x = Add(name='Input_Dec_GT_128')([x, x3])
        x = green_block(x, 128, name='Dec_GT_128', data_format=data_format)

        ### Green Block x1 (output filters=64)
        x = conv_3d_cls(
            filters=64,
            kernel_size=(1, 1, 1),
            strides=1,
            data_format=data_format,
            name='Dec_GT_ReduceDepth_64')(x)
        x = UpSampling3D(
            size=2,
            data_format=data_format,
            name='Dec_GT_UpSample_64')(x)
        x = Add(name='Input_Dec_GT_64')([x, x2])
        x = green_block(x, 64, name='Dec_GT_64', data_format=data_format)

        ### Green Block x1 (output filters=32)
        x = conv_3d_cls(
            filters=32,
            kernel_size=(1, 1, 1),
            strides=1,
            data_format=data_format,
            name='Dec_GT_ReduceDepth_32')(x)
        x = UpSampling3D(
            size=2,
            data_format=data_format,
            name='Dec_GT_UpSample_32')(x)
        x = Add(name='Input_Dec_GT_32')([x, x1])
        x = green_block(x, 32, name='Dec_GT_32', data_format=data_format)

        ### Blue Block x1 (output filters=32)
        x = conv_3d_cls(
            filters=32,
            kernel_size=(3, 3, 3),
            strides=1,
            padding='same',
            data_format=data_format,
            name='Input_Dec_GT_Output')(x)

        ### Output Block
        out_GT = Conv3D(
            filters=output_channels,  # No. of tumor classes is 3
            kernel_size=(1, 1, 1),
            strides=1,
            data_format=data_format,
            activation='sigmoid',
            name='Dec_GT_Output')(x)

    ## VAE (Variational Auto Encoder) Part
    # -------------------------------------------------------------------------

    ### VD Block (Reducing dimensionality of the data)
    x = GroupNormalization(groups=8, name='Dec_VAE_VD_GN', axis=1 if data_format == 'channels_first' else -1)(x4)
    x = Activation('relu', name='Dec_VAE_VD_relu')(x)
    x = conv_3d_cls(
        filters=16,
        kernel_size=(3, 3, 3),
        strides=2,
        padding='same',
        data_format=data_format,
        name='Dec_VAE_VD_Conv3D')(x)

    if shared_latent_space:
        # x1_flat = Flatten()(x1)
        # x2_flat = Flatten()(x2)
        x3_flat = Flatten()(x3)
        x4_flat = Flatten()(x4)

        # x1_dense = Dense(256, activation='relu')(x1_flat)
        # x2_dense = Dense(64, activation='relu')(x2_flat)
        x3_dense = Dense(256, activation='relu')(x3_flat)
        x4_dense = Dense(256, activation='relu')(x4_flat)

        x_dense_concat = Concatenate()([
            # x1_dense,
            # x2_dense,
            x3_dense,
            x4_dense],
        )
        x = Dense(dim_latent_space * 2, name='Dec_VAE_VD_Dense')(x_dense_concat)
    else:
        # Not mentioned in the paper, but the author used a Flattening layer here.
        x = Flatten(name='Dec_VAE_VD_Flatten')(x)
        x = Dense(dim_latent_space * 2, name='Dec_VAE_VD_Dense')(x)

    ### VDraw Block (Sampling)
    z_mean, z_log_var = x[:, :dim_latent_space], x[:, dim_latent_space:]
    x = Lambda(sampling, name='Dec_VAE_VDraw_Sampling')([z_mean, z_log_var])
    z_mean_z_log_var = x

    ### VU Block (Upsizing back to a depth of 256)
    x = Dense((H // 16) * (W // 16) * (D // 16))(x)
    x = Activation('relu')(x)
    if data_format == 'channels_first':
        x = Reshape((1, (H // 16), (W // 16), (D // 16)))(x)
    else:
        x = Reshape(((H // 16), (W // 16), (D // 16), 1))(x)
    x = conv_3d_cls(
        filters=256,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format=data_format,
        name='Dec_VAE_ReduceDepth_256')(x)
    x = UpSampling3D(
        size=2,
        data_format=data_format,
        name='Dec_VAE_UpSample_256')(x)

    ### Green Block x1 (output filters=128)
    x = conv_3d_cls(
        filters=128,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format=data_format,
        name='Dec_VAE_ReduceDepth_128')(x)
    x = UpSampling3D(
        size=2,
        data_format=data_format,
        name='Dec_VAE_UpSample_128')(x)
    x = green_block(x, 128, name='Dec_VAE_128', data_format=data_format)

    ### Green Block x1 (output filters=64)
    x = conv_3d_cls(
        filters=64,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format=data_format,
        name='Dec_VAE_ReduceDepth_64')(x)
    x = UpSampling3D(
        size=2,
        data_format=data_format,
        name='Dec_VAE_UpSample_64')(x)
    x = green_block(x, 64, name='Dec_VAE_64', data_format=data_format)

    ### Green Block x1 (output filters=32)
    x = conv_3d_cls(
        filters=32,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format=data_format,
        name='Dec_VAE_ReduceDepth_32')(x)
    x = UpSampling3D(
        size=2,
        data_format=data_format,
        name='Dec_VAE_UpSample_32')(x)
    x = green_block(x, 32, name='Dec_VAE_32', data_format=data_format)

    ### Blue Block x1 (output filters=32)
    x = conv_3d_cls(
        filters=32,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        data_format=data_format,
        name='Input_Dec_VAE_Output')(x)

    ### Output Block
    out_VAE = Conv3D(
        filters=c,
        kernel_size=(1, 1, 1),
        strides=1,
        activation='sigmoid' if z_score else None,
        data_format=data_format,
        name='Dec_VAE_Output')(x)

    if shared_latent_space:
        # x4 shape: 4, 4, 2, 256
        ### VU Block (Upsizing back to a depth of 256)
        x = Dense((H // 16) * (W // 16) * (D // 16))(z_mean)
        x = Activation('relu')(x)
        if data_format == 'channels_first':
            x = Reshape((1, (H // 16), (W // 16), (D // 16)))(x)
        else:
            x = Reshape(((H // 16), (W // 16), (D // 16), 1))(x)
        x = conv_3d_cls(
            filters=256,
            kernel_size=(1, 1, 1),
            strides=1,
            data_format=data_format,
            name='Dec_GT_ReduceDepth_256')(x)
        x = UpSampling3D(
            size=2,
            data_format=data_format,
            name='Dec_GT_UpSample_256')(x)

        ### Green Block x1 (output filters=128)
        x = conv_3d_cls(
            filters=128,
            kernel_size=(1, 1, 1),
            strides=1,
            data_format=data_format,
            name='Dec_GT_ReduceDepth_128')(x)
        x = UpSampling3D(
            size=2,
            data_format=data_format,
            name='Dec_GT_UpSample_128')(x)
        x = green_block(x, 128, name='Dec_GT_128', data_format=data_format)

        ### Green Block x1 (output filters=64)
        x = conv_3d_cls(
            filters=64,
            kernel_size=(1, 1, 1),
            strides=1,
            data_format=data_format,
            name='Dec_GT_ReduceDepth_64')(x)
        x = UpSampling3D(
            size=2,
            data_format=data_format,
            name='Dec_GT_UpSample_64')(x)
        x = green_block(x, 64, name='Dec_GT_64', data_format=data_format)

        ### Green Block x1 (output filters=32)
        x = conv_3d_cls(
            filters=32,
            kernel_size=(1, 1, 1),
            strides=1,
            data_format=data_format,
            name='Dec_GT_ReduceDepth_32')(x)
        x = UpSampling3D(
            size=2,
            data_format=data_format,
            name='Dec_GT_UpSample_32')(x)
        x = green_block(x, 32, name='Dec_GT_32', data_format=data_format)

        ### Blue Block x1 (output filters=32)
        x = conv_3d_cls(
            filters=32,
            kernel_size=(3, 3, 3),
            strides=1,
            padding='same',
            data_format=data_format,
            name='Input_Dec_GT_Output')(x)

        ### Output Block
        out_GT = Conv3D(
            filters=output_channels,
            kernel_size=(1, 1, 1),
            strides=1,
            activation='sigmoid',
            data_format=data_format,
            name='Dec_GT_Output')(x)

    def num_active_dims(y_true, y_pred):
        threshold = .1
        _num_active_dims = tf.math.count_nonzero(tf.exp(z_log_var) < threshold)

        return _num_active_dims

    # Build and Compile the model
    model = Model(inp, outputs=[out_GT, out_VAE, z_mean_z_log_var])
    model.compile(
        optimizer=adam(lr=adam_lr, decay=adam_decay),
        loss=[
            loss_gt(dice_e, data_format=data_format),
            lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred)),
            lambda y_true, y_pred: tf.reduce_mean(-.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))),
        ],
        loss_weights=[
            weight_dice,
            weight_L2,
            weight_KL,
        ],
        metrics={
            z_mean_z_log_var.name.split('/')[0]: num_active_dims,
        },
        experimental_run_tf_function=False,
    )

    return model


def get_vae_predictions(model, data):
    # https://keras.io/getting_started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction
    intermediate_layer_model = keras.Model(inputs=model.input, outputs=model.get_layer('Dec_VAE_VDraw_Mean').output)
    return intermediate_layer_model.predict(data)
