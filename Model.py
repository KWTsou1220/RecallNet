import tensorflow as tf
import numpy      as np

from ops import _weight_variable, _bias_variable
from RNN import *
from AttentionCell import *

class Model(object):
    def __init__(self, input_size, output_size, batch_size, time_step, LR, sequence_length, window=1):
        # basic setting
        self.input_size   = input_size
        self.output_size  = output_size
        self.time_step    = time_step
        self.batch_size   = batch_size
        self.LR           = LR
        self.sequence_length = sequence_length
        self.window       = window
                
        # placeholder: it allow to feed in different data in each iteration
        self.x  = tf.placeholder(tf.float32, [None, input_size*window], name='x') # [batch_size, time_step, input_dim]
        self.y1 = tf.placeholder(tf.float32, [None, output_size], name='y1')
        self.y2 = tf.placeholder(tf.float32, [None, output_size], name='y2')
        
        # feedforward
        with tf.variable_scope('MyModel'):
            self.feed_forward()
        
        # optimization
        self.compute_loss()
        self.optimizer = tf.train.AdamOptimizer(self.LR)
        
        grad_var       = self.optimizer.compute_gradients(self.loss)
        def GradientClip(grad):
            if grad is None:
                return grad
            #return tf.clip_by_norm(grad, 1)
            return tf.clip_by_value(grad, -1, 1)
        clip_grad_var = [(GradientClip(grad), var) for grad, var in grad_var ]
        self.train_op = self.optimizer.apply_gradients(clip_grad_var)
        
    def feed_forward(self):
        # Weight initial
        """
        Slv1_W1 = _weight_variable([self.input_size*self.window, 1000], name='l1_Slv1_W')
        Slv1_b1 = _bias_variable([1000, ], name='l1_Slv1_b')
        Slv1_fw_cell2 = LSTMCell(input_dim=1000, output_dim=400, name='l2_Slv1/ForwardLSTMCell')
        Slv1_bw_cell2 = LSTMCell(input_dim=1000, output_dim=400, name='l2_Slv1/BackwardLSTMCell')
        Slv1_W3 = _weight_variable([800, 600], name='l3_Slv1_W')
        Slv1_b3 = _bias_variable([600, ], name='l3_Slv1_b')
        Slv1_W41 = _weight_variable([600, self.output_size], name='l4_Slv1_W1')
        Slv1_W42 = _weight_variable([600, self.output_size], name='l4_Slv1_W2')
        Slv1_b41 = _bias_variable([self.output_size, ], name='l4_Slv1_b1')
        Slv1_b42 = _bias_variable([self.output_size, ], name='l4_Slv1_b2')
        """
        #"""
        Slv1_W1 = _weight_variable([self.input_size*self.window, 1000], name='l1_Slv1_W')
        Slv1_b1 = _bias_variable([1000, ], name='l1_Slv1_b')
        Slv1_fw_cell2 = LSTMCell(input_dim=1000, output_dim=400, name='l2_Slv1/ForwardLSTMCell')
        Slv1_bw_cell2 = LSTMCell(input_dim=1000, output_dim=400, name='l2_Slv1/BackwardLSTMCell')
        Slv1_W3 = _weight_variable([800, 600], name='l3_Slv1_W')
        Slv1_b3 = _bias_variable([600, ], name='l3_Slv1_b')
        Slv1_W41 = _weight_variable([600, self.output_size], name='l4_Slv1_W1')
        Slv1_W42 = _weight_variable([600, self.output_size], name='l4_Slv1_W2')
        Slv1_b41 = _bias_variable([self.output_size, ], name='l4_Slv1_b1')
        Slv1_b42 = _bias_variable([self.output_size, ], name='l4_Slv1_b2')
        
        Slv2_W1 = _weight_variable([self.input_size*self.window, 1000], name='l1_Slv2_W')
        Slv2_b1 = _bias_variable([1000, ], name='l1_Slv2_b')
        Slv2_fw_cell2 = LSTMCell(input_dim=1000, output_dim=400, name='l2_Slv2/ForwardLSTMCell')
        Slv2_bw_cell2 = LSTMCell(input_dim=1000, output_dim=400, name='l2_Slv2/BackwardLSTMCell')
        Slv2_W3 = _weight_variable([800, 600], name='l3_Slv2_W')
        Slv2_b3 = _bias_variable([600, ], name='l3_Slv2_b')
        Slv2_W41 = _weight_variable([600, self.output_size], name='l4_Slv2_W1')
        Slv2_W42 = _weight_variable([600, self.output_size], name='l4_Slv2_W2')
        Slv2_b41 = _bias_variable([self.output_size, ], name='l4_Slv2_b1')
        Slv2_b42 = _bias_variable([self.output_size, ], name='l4_Slv2_b2')
        #"""
        
        Mstr_W1 = _weight_variable([self.input_size*self.window, 1000], name='l1_Mstr_W')
        Mstr_b1 = _bias_variable([1000, ], name='l1_Mstr_b')
        Mstr_cell2 = LSTMCell(input_dim=1000, output_dim=800, name='l2_Mstr/LSTMCell')
        #Mstr_cell3 = BasicAttentionLSTMCell(
        #    input_dim=800, output_dim=700, attention_dim=500, mem_dim=800, name='l3_Mstr/AttentionLSTMCell')
        Mstr_cell3 = AttentionLSTMCell(input_dim=800, output_dim=700, attention_dim=500, mem_dim=[800, 800, 800], 
                                     name='l3_Mstr/AttentionLSTMCell')
        Mstr_W4 = _weight_variable([700, 600], name='l4_Mstr_W')
        Mstr_b4 = _bias_variable([600, ], name='l4_Mstr_b')
        Mstr_W51 = _weight_variable([600, self.output_size], name='l5_Mstr_W1')
        Mstr_W52 = _weight_variable([600, self.output_size], name='l5_Mstr_W2')
        Mstr_b51 = _bias_variable([self.output_size, ], name='l5_Mstr_b1')
        Mstr_b52 = _bias_variable([self.output_size, ], name='l5_Mstr_b2')
        
        # Feedforward
        """
        self.Slv1_z1 = tf.nn.bias_add( tf.matmul(self.x*self.window, Slv1_W1), Slv1_b1 )
        self.Slv1_z2 = BiRNN(Slv1_fw_cell2, Slv1_bw_cell2, tf.reshape(self.Slv1_z1, [-1, self.time_step, 1000]), 
                            sequence_length=self.sequence_length, init_state=Slv1_fw_cell2.zero_state(self.batch_size, dtype=tf.float32), 
                            name='l2_Slv1')
        self.Slv1_z3 = tf.nn.bias_add( tf.matmul(tf.reshape(self.Slv1_z2, [-1, 800]), Slv1_W3), Slv1_b3 )
        self.Slv1_z41 = tf.nn.bias_add( tf.matmul(self.Slv1_z3, Slv1_W41), Slv1_b41 )
        self.Slv1_z42 = tf.nn.bias_add( tf.matmul(self.Slv1_z3, Slv1_W42), Slv1_b42 )
        self.Slv1_summ = tf.add(tf.abs(self.Slv1_z41), tf.abs(self.Slv1_z42)) + (1e-6)
        self.Slv1_mask1 = tf.div(tf.abs(self.Slv1_z41), self.Slv1_summ)
        self.Slv1_mask2 = tf.div(tf.abs(self.Slv1_z42), self.Slv1_summ)
        self.Slv1_pred1 = tf.mul(self.x[:, ((self.window+1)/2-1)*self.input_size:((self.window+1)/2)*self.input_size], self.Slv1_mask1)
        self.Slv1_pred2 = tf.mul(self.x[:, ((self.window+1)/2-1)*self.input_size:((self.window+1)/2)*self.input_size], self.Slv1_mask2)
        """
        #"""
        self.Slv1_z1 = tf.nn.bias_add( tf.matmul(self.x*self.window, Slv1_W1), Slv1_b1 )
        self.Slv1_z2 = BiRNN(Slv1_fw_cell2, Slv1_bw_cell2, tf.reshape(self.Slv1_z1, [-1, self.time_step, 1000]), 
                            sequence_length=self.sequence_length, init_state=Slv1_fw_cell2.zero_state(self.batch_size, dtype=tf.float32), 
                            name='l2_Slv1')
        self.Slv1_z3 = tf.nn.bias_add( tf.matmul(tf.reshape(self.Slv1_z2, [-1, 800]), Slv1_W3), Slv1_b3 )
        self.Slv1_z41 = tf.nn.bias_add( tf.matmul(self.Slv1_z3, Slv1_W41), Slv1_b41 )
        self.Slv1_z42 = tf.nn.bias_add( tf.matmul(self.Slv1_z3, Slv1_W42), Slv1_b42 )
        self.Slv1_summ = tf.add(tf.abs(self.Slv1_z41), tf.abs(self.Slv1_z42)) + (1e-6)
        self.Slv1_mask1 = tf.div(tf.abs(self.Slv1_z41), self.Slv1_summ)
        self.Slv1_mask2 = tf.div(tf.abs(self.Slv1_z42), self.Slv1_summ)
        self.Slv1_pred1 = tf.mul(self.x[:, ((self.window+1)/2-1)*self.input_size:((self.window+1)/2)*self.input_size], self.Slv1_mask1)
        self.Slv1_pred2 = tf.mul(self.x[:, ((self.window+1)/2-1)*self.input_size:((self.window+1)/2)*self.input_size], self.Slv1_mask2)
        
        self.Slv2_z1 = tf.nn.bias_add( tf.matmul(self.x*self.window, Slv2_W1), Slv2_b1 )
        self.Slv2_z2 = BiRNN(Slv2_fw_cell2, Slv2_bw_cell2, tf.reshape(self.Slv2_z1, [-1, self.time_step, 1000]), 
                            sequence_length=self.sequence_length, init_state=Slv2_fw_cell2.zero_state(self.batch_size, dtype=tf.float32), 
                            name='l2_Slv2')
        self.Slv2_z3 = tf.nn.bias_add( tf.matmul(tf.reshape(self.Slv2_z2, [-1, 800]), Slv2_W3), Slv2_b3 )
        self.Slv2_z41 = tf.nn.bias_add( tf.matmul(self.Slv2_z3, Slv2_W41), Slv2_b41 )
        self.Slv2_z42 = tf.nn.bias_add( tf.matmul(self.Slv2_z3, Slv2_W42), Slv2_b42 )
        self.Slv2_summ = tf.add(tf.abs(self.Slv2_z41), tf.abs(self.Slv2_z42)) + (1e-6)
        self.Slv2_mask1 = tf.div(tf.abs(self.Slv2_z41), self.Slv2_summ)
        self.Slv2_mask2 = tf.div(tf.abs(self.Slv2_z42), self.Slv2_summ)
        self.Slv2_pred1 = tf.mul(self.x[:, ((self.window+1)/2-1)*self.input_size:((self.window+1)/2)*self.input_size], self.Slv2_mask1)
        self.Slv2_pred2 = tf.mul(self.x[:, ((self.window+1)/2-1)*self.input_size:((self.window+1)/2)*self.input_size], self.Slv2_mask2)
        #"""
        
        self.Mstr_z1 = tf.nn.bias_add( tf.matmul(self.x, Mstr_W1), Mstr_b1 )
        self.Mstr_z2 = RNN(Mstr_cell2, tf.reshape(self.Mstr_z1, [-1, self.time_step, 1000]), sequence_length=self.sequence_length, 
                          init_state=Mstr_cell2.zero_state(self.batch_size, dtype=tf.float32), name='l2_Mstr')
        #self.Mstr_z3 = BasicAttentionRNN(Mstr_cell3, self.Mstr_z2, self.Slv1_z2, sequence_length=self.sequence_length, 
        #                           init_state=Mstr_cell3.zero_state(self.batch_size, dtype=tf.float32), name='l3_Mstr')
        self.Mstr_z3 = AttentionRNN(cell=Mstr_cell3, x=self.Mstr_z2, M=[self.Mstr_z2, self.Slv1_z2, self.Slv2_z2], 
                                                   sequence_length=self.sequence_length, 
                                                   init_state=Mstr_cell3.zero_state(self.batch_size, dtype=tf.float32), name='l3_Mstr')
        self.Mstr_z4 = tf.nn.bias_add( tf.matmul(tf.reshape(self.Mstr_z3, [-1, 700]), Mstr_W4), Mstr_b4 )
        self.Mstr_z51 = tf.nn.bias_add( tf.matmul(self.Mstr_z4, Mstr_W51), Mstr_b51 )
        self.Mstr_z52 = tf.nn.bias_add( tf.matmul(self.Mstr_z4, Mstr_W52), Mstr_b52 )
        self.Mstr_summ = tf.add(tf.abs(self.Mstr_z51), tf.abs(self.Mstr_z52)) + (1e-6)
        self.Mstr_mask1 = tf.div(tf.abs(self.Mstr_z51), self.Mstr_summ)
        self.Mstr_mask2 = tf.div(tf.abs(self.Mstr_z52), self.Mstr_summ)
        self.Mstr_pred1 = tf.mul(self.x[:, ((self.window+1)/2-1)*self.input_size:((self.window+1)/2)*self.input_size], self.Mstr_mask1)
        self.Mstr_pred2 = tf.mul(self.x[:, ((self.window+1)/2-1)*self.input_size:((self.window+1)/2)*self.input_size], self.Mstr_mask2)
    
    def compute_loss(self):
        self.Slv1_loss = (self.ms_error(self.Slv1_pred1, self.y1)+self.ms_error(self.Slv1_pred2, self.y2))/2
        self.Mstr_loss = (self.ms_error(self.Mstr_pred1, self.y1)+self.ms_error(self.Mstr_pred2, self.y2))/2
        self.loss     = (0.5*self.Slv1_loss + 0.5*self.Mstr_loss)
        
    def ms_error(self, y_pre, y_target):
        return tf.reduce_sum(tf.reduce_sum( tf.square(tf.sub(y_pre, y_target)), 1))
    