import gin
import tensorflow as tf
from tensorflow.python.keras.utils.version_utils import training

from layers import vgg_block

@gin.configurable
def vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """Defines a VGG-like architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate

    Returns:
        (keras.Model): keras model object
    """
    # Load the pretrained VGG16 model excluding the top classification layer
    base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    assert n_blocks > 0, 'Number of blocks has to be at least 1.'

    inputs = tf.keras.Input(input_shape)
    out = base_model(inputs, training=False) # training = false , to avoid updating batchnorm
    for i in range(1, n_blocks):
        out = vgg_block(out, base_filters * 2 ** (i), kernel_size=(3,3))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes-1, activation = tf.sigmoid)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')