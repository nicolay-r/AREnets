"""
(c) Ilya Ivanov
github: https://github.com/ilivans
code: https://github.com/ilivans/tf-rnn-attention
"""

import tensorflow as tf


def self_attention_by_z_yang(inputs, attention_size, time_major=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article

    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
    Returns:
        alphas: Tensor
            tensor of shape (B, T), which corresponds to weights of the particular time-series item.
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    initializer = tf.random_normal_initializer(stddev=0.1)

    # Trainable parameters
    w_omega = tf.compat.v1.get_variable(name="w_omega", shape=[hidden_size, attention_size], initializer=initializer)
    b_omega = tf.compat.v1.get_variable(name="b_omega", shape=[attention_size], initializer=initializer)
    u_omega = tf.compat.v1.get_variable(name="u_omega", shape=[attention_size], initializer=initializer)

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    return alphas


def calculate_sequential_attentive_weights_by_z_yang(inputs, alphas):
    """Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
       The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
       for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
       Variables notation is also inherited from the article

        Proposes to reduce the time-series parameter
        inputs: Tensor
            tensor of shape (B, T, D), where
                B -- batch, T -- time-series (context terms), D -- data or hidden embedding.
        alphas: Tensor
            tensor of shape (B, T), which corresponds to weights of the particular time-series item.
    """

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    return output
