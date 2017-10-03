import tensorflow as tf
import numpy      as np

def RNN(cell, x, sequence_length, init_state, name):
    """
    This function construct simple framework of recurrent neural network.
    Input:
        cell: RNN cell object
        x   : input of shape [batch_size, time_step, input_dim]
        sequence_length: list of sequence length corresponding to each batch [time_step1, time_step2, ...]
        init_state: inital state of RNN
    Output:
        H: representation of input of shape [batch_size, time_step, input_dim]
    """
    with tf.variable_scope(name or "RNN", reuse=True):
        # Get shape
        time_step, input_dim = x.get_shape().as_list()[1:]
        batch_size = init_state[0].get_shape().as_list()[0]
        if batch_size is None: # initial phase
            batch_size = 2
        
        # RNN feedforward
        H = [] # all the hidden states of RNN
        h = init_state # current hidden state
        for t in xrange(time_step):
            o, h = cell(x[:, t, :], h)
            H.append(o)
        H = tf.pack(H, axis=2)
        H = tf.transpose(H, perm=[0, 2, 1]) # H is a tensor of shape [batch_size, time_step, state_size]

        # Cutting the sequence by sequence_length
        state_size = H.get_shape().as_list()[2]
        mask = np.zeros(shape=[batch_size, time_step, state_size]) # [batch_size, time_step, state_size]
        for b in xrange(batch_size):
            mask[b, 0:sequence_length[b], :] = 1
        H = H*mask
    
    return H
    
def BiRNN(fw_cell, bw_cell, x, sequence_length, init_state, name):
    """
    This function construct a framework of bi-directional recurrent neural network.
    Input:
        fw_cell: RNN cell object of forward propagation
        bw_cell: RNN cell object of backward propagation
        x   : input of shape [batch_size, time_step, input_dim]
        sequence_length: list of sequence length corresponding to each batch [time_step1, time_step2, ...]
        init_state: inital state of RNN
    Output:
        H: representation of input of shape [batch_size, time_step, input_dim]
    """
    with tf.variable_scope(name or "BiRNN", reuse=True):
        # Get shape
        time_step, input_dim = x.get_shape().as_list()[1:]
        batch_size = init_state[0].get_shape().as_list()[0]
        if batch_size is None: # initial phase
            batch_size = 2
        
        # Forward RNN
        fw_H = [] # all the forward hidden states
        fw_h = init_state # the current forward hidden states
        for t in xrange(time_step):
            fw_o, fw_h = fw_cell(x[:, t, :], fw_h, name='ForwardLSTMCell')
            fw_H.append(fw_o)
        fw_H = tf.pack(fw_H, axis=2)
        fw_H = tf.transpose(fw_H, perm=[0, 2, 1]) # fw_H is a tensor of shape [batch_size, time_step, state_size]
        
        # Cutting the redundant sequence 
        state_size = fw_H.get_shape().as_list()[2]
        mask = np.zeros(shape=[batch_size, time_step, state_size]) # [batch_size, time_step, state_size]
        for b in xrange(batch_size):
            mask[b, 0:sequence_length[b], :] = 1
        fw_H = fw_H*mask
        
        # Backward RNN
        bw_H = [] # all the backward hidden states
        bw_h = init_state # the current backward hidden state
        for t in reversed(xrange(time_step)):
            bw_o, bw_h = bw_cell(x[:, t, :], bw_h, name='BackwardLSTMCell')
            bw_H.append(bw_o)
        bw_H = bw_H[::-1] # reverse
        bw_H = tf.pack(bw_H, axis=2)
        bw_H = tf.transpose(bw_H, perm=[0, 2, 1]) # bw_H is a tensor of shape [batch_size, time_step, state_size]
        
        # Cutting the sequence
        bw_H = bw_H*mask
        
        # Merge fw_H and bw_H
        H = tf.concat(concat_dim=2, values=[fw_H, bw_H]) # [batch_size, time_step, state_size*2]
        
    return H

def AttentionRNN(cell, x, M, sequence_length, init_state, name):
    """
    This function construct a framework of recurrent neural network with attention mechanism. AttentionRNN will attend over both encoder 
    memory M1 and decoder memory M2. 
    Input:
        cell: RNNCell object
        x   : input of shape [batch_size, time_step, input_dim]
        M   : list of external memory, each element has shape [batch_size, time_step, mem1_dim]
        M1  : external memory of shape [batch_size, time_step, mem1_dim]
        M2  : self memory of shape [batch_size, time_step, mem2_dim]
        sequence_length: list of sequence length corresponding to each batch [time_step1, time_step2, ...]
        init_state: inital state of RNN
    Output:
        H: hidden state of shape [batch_size, time_step, state_dim]
    """
    with tf.variable_scope(name or "AttentionRNN", reuse=True):
        # Get shape
        time_step, input_dim = x.get_shape().as_list()[1:]
        batch_size = init_state[0].get_shape().as_list()[0]
        if batch_size is None: # initial phase
            batch_size = 2
        
        # RNN feedforward
        H = [] # all the hidden states
        a1, a2, c1, c2 = [], [], [], []
        h = init_state # the current hidden state
        for t in xrange(time_step):
            o, h, att = cell(x[:, t, :], h, M)
            H.append(o)
            #a1.append(att[0]) # [batch_size, mem_dim]
            #a2.append(att[1])
            #c1.append(att[2])
            #c2.append(att[3])
        H = tf.pack(H, axis=2)
        H = tf.transpose(H, perm=[0, 2, 1]) # H is a tensor of shape [batch_size, time_step, state_size]
        #a1 = tf.pack(a1, axis=2) # attention weights for encoder memory
        #a2 = tf.pack(a2, axis=2) # attention weights for decoder memory
        #c1 = tf.pack(c1, axis=2)
        #c1 = tf.transpose(c1, perm=[0, 2, 1])
        #c1 = tf.reshape(c1, [-1, 800])
        #c2 = tf.pack(c2, axis=2)
        #c2 = tf.transpose(c2, perm=[0, 2, 1])
        #c2 = tf.reshape(c2, [-1, 800])
        
        # Cutting the redundant sequence 
        state_size = H.get_shape().as_list()[2]
        mask = np.zeros(shape=[batch_size, time_step, state_size]) # [batch_size, time_step, state_size]
        for b in xrange(batch_size):
            mask[b, 0:sequence_length[b], :] = 1
        H = H*mask
    
    return H #, (a1, a2, c1, c2)

def BasicAttentionRNN(cell, x, M, sequence_length, init_state, name):
    """
    This function construct a framework of recurrent neural network with attention mechanism. BasicAttentionRNN will attend over only 
    decoder memory M.
    Input:
        cell: RNNCell object
        x   : input of shape [batch_size, time_step, input_dim]
        M   : external memory of shape [batch_size, time_step, mem1_dim]
        sequence_length: list of sequence length corresponding to each batch [time_step1, time_step2, ...]
        init_state: inital state of RNN
    Output:
        H: hidden state of shape [batch_size, time_step, state_dim]
    """
    with tf.variable_scope(name or "AttentionRNN", reuse=True):
        # Get shape
        time_step, input_dim = x.get_shape().as_list()[1:]
        batch_size = init_state[0].get_shape().as_list()[0]
        if batch_size is None: # initial phase
            batch_size = 2
        
        # RNN feedforward
        H = [] # all the hidden states
        h = init_state # the current hidden state
        for t in xrange(time_step):
            o, h = cell(x[:, t, :], h, M)
            H.append(o)
        H = tf.pack(H, axis=2)
        H = tf.transpose(H, perm=[0, 2, 1]) # H is a tensor of shape [batch_size, time_step, state_size]

        # Cutting the redundant sequence 
        state_size = H.get_shape().as_list()[2]
        mask = np.zeros(shape=[batch_size, time_step, state_size]) # [batch_size, time_step, state_size]
        for b in xrange(batch_size):
            mask[b, 0:sequence_length[b], :] = 1
        H = H*mask
    
    return H
