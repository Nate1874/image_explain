import tensorflow as tf


def conv(inputs, out_num, kernel_size, stride, padding, scope,training, norm=True):
    outs = tf.layers.conv2d(
        inputs, out_num, kernel_size, strides =stride, padding=padding, name=scope+'/conv',
        kernel_initializer=tf.truncated_normal_initializer)
    if norm:
        return tf.contrib.layers.batch_norm(
            outs, decay=0.9, epsilon=1e-5, activation_fn=tf.nn.relu, is_training = training,
            updates_collections=tf.GraphKeys.UPDATE_OPS, scope=scope+'/batch_norm')
    else:
        return outs


def deconv(inputs, out_num, kernel_size, scope, training):
    outs = tf.layers.conv2d_transpose(
            inputs, out_num, kernel_size, (2, 2), padding='same', name=scope,
            kernel_initializer=tf.truncated_normal_initializer)
    outs = tf.contrib.layers.batch_norm(
        outs, decay=0.9, epsilon=1e-5, activation_fn=tf.nn.relu,is_training = training,
        updates_collections=tf.GraphKeys.UPDATE_OPS, scope=scope+'/batch_norm')
    return outs


