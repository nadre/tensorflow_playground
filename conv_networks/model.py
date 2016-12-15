import layers
import tensorflow as tf

def get_model(data, params):
    with tf.variable_scope("conv1"):
        kernel_shape = [ params["patch_size"],
                         params["patch_size"],
                         params["num_channels"],
                         params["depth"]
                    ]
        bias_shape = params["depth"]
        stride = 2
        conv1 = layers.conv_relu(data, kernel_shape, bias_shape, stride)

    with tf.variable_scope("conv2"):
        kernel_shape = [ params["patch_size"],
                         params["patch_size"],
                         params["depth"],
                         params["depth"]
                    ]
        bias_shape = params["depth"]
        stride = 2
        conv2 = layers.conv_relu(conv1, kernel_shape, bias_shape, stride)

    with tf.variable_scope("reshape"):
        shape = conv2.get_shape().as_list()
        reshaped = tf.reshape(conv2, [shape[0], shape[1] * shape[2] * shape[3]])

    with tf.variable_scope("relu1"):
        relu1 = layers.relu(reshaped, params["num_hidden"])

    with tf.variable_scope("relu2"):
        relu2 = layers.relu(relu1, params["num_labels"])

    return relu2
