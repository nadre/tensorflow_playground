import tensorflow as tf

def conv_relu(data, kernel_shape, bias_shape, stride):

    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())

    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))

    conv = tf.nn.conv2d(data, weights,
        strides=[1, stride, stride, 1], padding='SAME')

    conv_relu = tf.nn.relu(conv + biases)

    return conv_relu

def relu(data, output_shape):
    input_shape = [ data.get_shape().as_list()[1], output_shape]
    weights = tf.get_variable("weights", input_shape,
        initializer=tf.random_normal_initializer())

    biases = tf.get_variable("biases", output_shape,
        initializer=tf.constant_initializer(0.0))

    mul = tf.matmul(data, weights)

    return tf.nn.relu(mul + biases)
