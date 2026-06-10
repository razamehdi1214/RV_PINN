import tensorflow as tf


def stress_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))
