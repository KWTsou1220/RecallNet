import tensorflow as tf

def _weight_variable(shape, name='weights', initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)):
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

def _bias_variable(shape, name='biases'):
    initializer = tf.constant_initializer(0.1)
    return tf.get_variable(name=name, shape=shape, initializer=initializer) 

def linear(inputs, output_dim, name=None):
    with tf.variable_scope(name or "linear"):
        # Get shapes
        input_dim = 0
        for i in inputs:
            input_dim += i.get_shape().as_list()[1]
        
        # Get variable
        initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        W = _weight_variable(shape=[input_dim, output_dim], initializer=initializer)
        b = _bias_variable(shape=[output_dim, ])
        
        return tf.matmul(tf.concat_v2(inputs, 1), W) + b

def sequence_mask(batch_size, time_step, input_dim, sequence_length):
    """
    Generate the sequence masking for dynamic rnn with variable length
    """
    indices = []
    values = []
    for b in xrange(batch_size):
        tmp = [[b, i, j] for i in xrange(sequence_length[b]) for j in xrange(input_dim)]
        indices = indices + tmp
        values = values + [1]
    shape = [batch_size, time_step, input_dim]
    delta = tf.SparseTensor(indices, values, shape)
    result = tf.sparse_tensor_to_dense(delta)
    
    return result
    