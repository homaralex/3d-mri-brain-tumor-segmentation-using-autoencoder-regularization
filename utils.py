import tensorflow as tf
import tensorflow.python.keras.backend as K


# From keras-team/keras/blob/master/examples/variational_autoencoder.py
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var, deterministic_val_pas = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = tf.random.normal(shape=(batch, dim))
    if deterministic_val_pas:
        training = K.learning_phase()
        epsilon = epsilon * training

    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
