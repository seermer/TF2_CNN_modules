import tensorflow as tf
from tensorflow.keras import backend as K


def leaky_relu6(alpha=0.1):
    @tf.function
    def leaky_relu6_inner(x):
        if x.dtype.is_integer:
            x = tf.cast(x, tf.float64)
        return K.relu(x, alpha=alpha, max_value=6.)

    return leaky_relu6_inner


def double_leaky_relu6(alpha_lower=0.1, alpha_upper=0.1):
    @tf.function
    def double_leaky_relu6_inner(x):
        if x.dtype.is_integer:
            x = tf.cast(x, tf.float64)
        return K.minimum(K.relu(x, alpha=alpha_lower), tf.add(tf.multiply(tf.add(x, -6.), alpha_upper), 6.))

    return double_leaky_relu6_inner


@tf.function
def hardswish(x):
    if x.dtype.is_integer:
        x = tf.cast(x, tf.float64)
    return tf.where(tf.less_equal(x, tf.constant(-3., dtype=x.dtype)), tf.constant(0., dtype=x.dtype),
                    tf.where(tf.greater_equal(x, tf.constant(3., dtype=x.dtype)), x,
                             tf.multiply(x, tf.divide(tf.add(x, 3.), 6.))))
