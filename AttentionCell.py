import numpy      as np
import tensorflow as tf

from ops import _weight_variable, _bias_variable, linear

class LSTMCell(tf.nn.rnn_cell.RNNCell):
    """
    Implementation of LSTM cell with attention mechanism
    """
    def __init__(self, input_dim, output_dim, name=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        x = tf.ones(shape=[2, input_dim])
        prev_h = (tf.ones(shape=[2, output_dim]), tf.ones(shape=[2, output_dim]))
        self.__call__(x, prev_h, name)
        
    @property
    def state_size(self):
        return (self.output_dim, self.output_dim)
    
    @property
    def output_size(self):
        return self.output_dim
    
    @property
    def input_size(self):
        return self.input_dim
    
    def __call__(self, x, prev_h=None, name=None):
        """
        Input:
            x: input of shape [batch_size, input_dim]
            M1: memory 1 of shape [batch_size, time_step, mem1_dim]
            M2: memory 2 of shape [batch_size, time_step, mem2_dim]
            prev_h: tuple of previous state [out, cell] each of shape [batch_size, output_dim]
        Output:
            out: output of the cell
            h: tuple of hidden state [out, cell]
        """
        # Get shape
        batch_size = x.get_shape().as_list()[0]
        
        with tf.variable_scope(name or "LSTMCell"):
            def new_gate(gate_name):
                return linear([x, prev_h[0]], output_dim=self.output_dim, name=gate_name)
                
            # input, forget, and output gates
            i = tf.sigmoid(new_gate("input_gate"))
            f = tf.sigmoid(new_gate("forget_gate"))
            o = tf.sigmoid(new_gate("output_gate"))
            u = tf.tanh(new_gate("update"))
            
            # update the state of the LSTM
            cell = tf.add_n([f*prev_h[1], i*u])
            out = o*tf.tanh(cell)
            
        return out, (out, cell)

class BasicAttentionLSTMCell(tf.nn.rnn_cell.RNNCell):
    """
    Implementation of LSTM cell with attention mechanism over only decoder memory
    """
    def __init__(self, input_dim, output_dim, attention_dim, mem_dim, name=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.attention_dim = attention_dim
        self.mem_dim = mem_dim
        
        x = tf.ones(shape=[2, input_dim])
        prev_h = (tf.ones(shape=[2, output_dim]), tf.ones(shape=[2, output_dim]))
        M = tf.ones(shape=[2, 2, mem_dim])
        self.__call__(x, prev_h, M, name)
        
    @property
    def state_size(self):
        return (self.output_dim, self.output_dim)
    
    @property
    def output_size(self):
        return self.output_dim
    
    @property
    def input_size(self):
        return self.input_dim
    
    def __call__(self, x, prev_h, M, name=None):
        """
        Input:
            x: input of shape [batch_size, input_dim]
            M: memory 1 of shape [batch_size, time_step, mem1_dim]
            prev_h: tuple of previous state [out, cell] each of shape [batch_size, output_dim]
        Output:
            out: output of the cell
            h: tuple of hidden state [out, cell]
        """
        # Get shape
        batch_size, time_step, mem_dim = M.get_shape().as_list()
        
        with tf.variable_scope(name or "AttentionLSTMCell"):
            # compute attention weight and context
            e = self.attention_weight(prev_h[0], M, name="attention_weight")
            c = tf.einsum('ijk,ij->ik', M, e) # [batch_size, mem_dim]
            
            # compute gate
            g = linear([prev_h[0], c], output_dim=mem_dim, name="gate")
            c = tf.sigmoid(g)*c
            
            
            def new_gate(gate_name):
                return linear([x, prev_h[0], c], output_dim=self.output_dim, name=gate_name)
                
            # input, forget, and output gates
            i = tf.sigmoid(new_gate("input_gate"))
            f = tf.sigmoid(new_gate("forget_gate"))
            o = tf.sigmoid(new_gate("output_gate"))
            u = tf.tanh(new_gate("update"))
            
            # update the state of the LSTM
            cell = tf.add_n([f*prev_h[1], i*u])
            out = o*tf.tanh(cell)
            
        return out, (out, cell)
    
    def attention_weight(self, prev_out, M, name=None):
        """
        Input:
            prev_out: previous hidden state of shape [batch_size, state_dim]
            M: memory of shape [batch_size, time_step, mem_dim]
        """
        with tf.variable_scope(name or "attention_weight"):
            # Get shape
            batch_size, state_dim = prev_out.get_shape().as_list()
            _, time_step, mem_dim = M.get_shape().as_list()
            
            
            # Get variable
            initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
            W = _weight_variable(shape=[state_dim, self.attention_dim], initializer=initializer, name="w_prev_out")
            U = _weight_variable(shape=[mem_dim, self.attention_dim], initializer=initializer, name="w_mem")
            v = _weight_variable(shape=[self.attention_dim, 1], initializer=initializer, name="v")
            
            # Computing Attention Weight
            X1 = tf.matmul(prev_out, W) # [batch_size, attention_dim]
            X1 = tf.expand_dims(input=X1, axis=1) # [batch_size, 1, attention_dim]
            X2 = tf.einsum('ijk,kl->ijl', M, U) # [batch_size, time_step, attention_dim]
            e  = tf.einsum('ijk,kl->ij', tf.tanh(X1+X2), v) # [batch_size, time_step]
            e  = tf.nn.softmax(e) 
            
        return e # [batch_size, time_step]
    
class AttentionLSTMCell(BasicAttentionLSTMCell):
    """
    Implementation of LSTM cell with attention mechanism over both encoder memory and decoder memory
    """
    def __init__(self, input_dim, output_dim, attention_dim, mem_dim, name=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.attention_dim = attention_dim
        self.mem_dim = mem_dim
        
        x = tf.ones(shape=[2, input_dim])
        prev_h = (tf.ones(shape=[2, output_dim]), tf.ones(shape=[2, output_dim]))
        M = []
        for idx in xrange(len(mem_dim)):
            tmp_M = tf.ones(shape=[2, 2, mem_dim[idx]])
            M.append(tmp_M)
        self.__call__(x, prev_h, M, name)
        
    @property
    def state_size(self):
        return (self.output_dim, self.output_dim)
    
    @property
    def output_size(self):
        return self.output_dim
    
    @property
    def input_size(self):
        return self.input_dim
    
    def __call__(self, x, prev_h, M, name=None):
        """
        Input:
            x: input of shape [batch_size, input_dim]
            M: list of memory 1 of shape [batch_size, time_step, mem_dim]
            prev_h: tuple of previous state [out, cell] each of shape [batch_size, output_dim]
        Output:
            out: output of the cell
            h: tuple of hidden state [out, cell]
        """
        # Get shape
        mem_num = len(M)
        mem_dim = []
        for idx in xrange(mem_num):
            batch_size, time_step, tmp = M[idx].get_shape().as_list()
            mem_dim.append(tmp)
        
        with tf.variable_scope(name or "AttentionLSTMCell"):
            e = []
            c = []
            g = []
            for idx in xrange(mem_num):
                # compute attention weight and context
                tmp_e = self.attention_weight(prev_h[0], M[idx], name="attention_weight"+str(idx)) # [batch_size, time_step]
                e.append(tmp_e)
                tmp_c = tf.einsum('ijk,ij->ik', M[idx], e[idx]) # [batch_size, mem_dim]
                c.append(tmp_c)
                
                # compute gate
                tmp_g = linear([prev_h[0], c[idx]], output_dim=mem_dim[idx], name="gate"+str(idx))
                g.append(tmp_g)
                c[idx] = tf.sigmoid(g[idx])*c[idx]
                
            """
            e1 = self.attention_weight(prev_h[0], M1, name="attention_weight1") # [batch_size, time_step]
            e2 = self.attention_weight(prev_h[0], M2, name="attention_weight2") # [batch_size, time_step]
            c1 = tf.einsum('ijk,ij->ik', M1, e1) # [batch_size, mem_dim]
            c2 = tf.einsum('ijk,ij->ik', M2, e2) # [batch_size, mem_dim]
            
            # compute gate
            g1 = linear([prev_h[0], c1], output_dim=mem1_dim, name="gate1")
            c1 = tf.sigmoid(g1)*c1
            g2 = linear([prev_h[0], c2], output_dim=mem2_dim, name="gate2")
            c2 = tf.sigmoid(g2)*c2
            """
            
            def new_gate(gate_name):
                return linear([x, prev_h[0]]+c, output_dim=self.output_dim, name=gate_name)
                
            # input, forget, and output gates
            i = tf.sigmoid(new_gate("input_gate"))
            f = tf.sigmoid(new_gate("forget_gate"))
            o = tf.sigmoid(new_gate("output_gate"))
            u = tf.tanh(new_gate("update"))
            
            # update the state of the LSTM
            cell = tf.add_n([f*prev_h[1], i*u])
            out = o*tf.tanh(cell)
            
        return out, (out, cell), e
            
         
        
        
        
        
        
        
        
        
        