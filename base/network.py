# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 13:15:17 2018

@author: jwm
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
from six import string_types,iteritems
import numpy as np
from utils.anchors import proposal_layer as proposal_layer_np
from utils.anchors import anchor_target_layer as anchor_target_layer_np

def layer(op):
    '''
       Decorator for composable network layers.
    '''
    def layer_decorated(self, *args, **kwargs):
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__)) # Automatically set a name if not provided.
        # Figure out the layer inputs.
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        
        layer_output = op(self, layer_input, *args, **kwargs) # Perform the operation and get the output.
        self.layers[name] = layer_output                      # Add to layer LUT.
        self.feed(layer_output)                               # This output is now the input for the next layer.
        return self                                           # Return self for chained calls.
    return layer_decorated

class Network(object):

    def __init__(self, inputs):
        
        self.inputs = []            # The input nodes for this network
        self.layers = dict(inputs)      # Mapping from layer names to layers
        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')
        
    def load_npz(self, weight_path, sess, ignore_missing=False):
        '''
           Load network weights.
           sess: The current TensorFlow session
           ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(weight_path, encoding='latin1').item() #pylint: disable=no-member

        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in iteritems(data_dict[op_name]):
                    try:
                        var = tf.get_variable(param_name)
                        sess.run(var.assign(data))
                    except ValueError:
                        print("ignore " + param_name)
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''
            Set the input(s) for the next operation by replacing the terminal nodes.
            The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.inputs = []
        for fed_layer in args:
            if isinstance(fed_layer, string_types):
                try:
                    fed_layer = self.layers[fed_layer]
#                    print(fed_layer)
                except KeyError:
                    print (list(self.layers.keys()))
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.inputs.append(fed_layer)
        return self

    def get_final_output(self):
        '''
            Returns the final network output.
        '''
        return self.inputs[-1]
        
    def get_output(self, layer):
        '''
            Returns the appointed network output.
        '''
        try:
            layer = self.layers[layer]
        except KeyError:
            print(list(self.layers.keys()))
            raise KeyError('Unknown layer name fed: %s' % layer)
        return layer

    def get_unique_name(self, prefix):
        '''
            Returns an index-suffixed unique name for the given prefix.
            This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        '''
            Creates a new TensorFlow variable.
        '''
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        '''
            Verifies that the padding is one of the supported ones.
        '''
        assert padding in ('SAME', 'VALID')
        
    def l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                                 dtype=tensor.dtype.base_dtype,
                                                 name='weight_decay')
                # return tf.mul(l2_weight, tf.nn.l2_loss(tensor), name='value')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
        return regularizer

        
    @layer
    def Bilstm(self, _input, d_i, d_h, d_o, name, weight_decay=0.0005, trainable=True):
        '''parameter
           _input:  input
           d_i:     dimension of input layer
           d_h:     dimension of hidden layer or number of cells
           d_o:     dimension of ouput layer
        '''
        with tf.variable_scope(name):
            shape = tf.shape(_input)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            _input = tf.reshape(_input, [N * H, W, C])
            _input.set_shape([None, None, d_i]) # (time_step_size, batch_size, input_vec_size)
            # 单层双向动态RNN
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple=True)
            lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,_input,dtype=tf.float32)
            lstm_out = tf.concat(lstm_out, axis=-1)
            lstm_out = tf.reshape(lstm_out, [N * H * W, 2 * d_h])

            init_weights = tf.truncated_normal_initializer(stddev=0.1)
            init_biases = tf.constant_initializer(0.0)
            
            weights = self.make_var('weights', [2 * d_h, d_o], init_weights, trainable,
                                    regularizer=self.l2_regularizer(weight_decay))
                                    
            biases = self.make_var('biases', [d_o], init_biases, trainable)
            
            outputs = tf.matmul(lstm_out, weights) + biases
            outputs = tf.reshape(outputs, [N, H, W, d_o])
            return outputs

    @layer
    def lstm(self, _input, d_i, d_h, d_o, name, weight_decay=0.0005, trainable=True):
        '''parameter
           _input:  input
           d_i:     dimension of input layer
           d_h:     dimension of hidden layer or number of cells
           d_o:     dimension of ouput layer
        reference: http://www.360doc.com/content/17/0321/10/10408243_638692495.shtml
        '''
        with tf.variable_scope(name):
            shape = tf.shape(_input)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            _input = tf.reshape(_input, [N * H, W, C])
            _input.set_shape([None, None, d_i]) # (time_step_size, batch_size, input_vec_size)

            lstm_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple=True)
            initial_state = lstm_cell.zero_state(N * H, dtype=tf.float32)
            lstm_out, last_state = tf.nn.dynamic_rnn(lstm_cell, _input,initial_state=initial_state, dtype=tf.float32)

            lstm_out = tf.reshape(lstm_out, [N * H * W, d_h])

            init_weights = tf.truncated_normal_initializer(stddev=0.1)
            init_biases = tf.constant_initializer(0.0)
            
            weights = self.make_var('weights', [d_h, d_o], init_weights, trainable,
                                    regularizer=self.l2_regularizer(weight_decay))
            biases = self.make_var('biases', [d_o], init_biases, trainable)
            
            outputs = tf.matmul(lstm_out, weights) + biases
            outputs = tf.reshape(outputs, [N, H, W, d_o])
            return outputs

    @layer
    def conv(self,_input,k_h,k_w,c_o,s_h,s_w,name,relu=True,padding='SAME',biased=True):
        """ parameters
            inp: input images
            k_h: kernal height
            k_w: kernal width
            c_o: output channel
            s_h: stride height
            s_w: stride width
        """
        self.validate_padding(padding)       # Verify that the padding is acceptable
        c_i = int(_input.get_shape()[-1])       # Get the number of channels in the input
        # Verify that the grouping parameter is valid
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)  # Convolution for a given input and kernel
        with tf.variable_scope(name) as scope:
            init_weights = tf.truncated_normal_initializer(0.0,stddev=0.01)
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i, c_o], initializer=init_weights)
            output = convolve(_input, kernel)        # This is the common-case. Convolve the input without any further complications.
            # Add the biases
            if biased:
                init_biases = tf.constant_initializer(0.0)
                biases = self.make_var('biases', shape=[c_o], initializer=init_biases)
                output = tf.nn.bias_add(output, biases) # default name: "BiasAdd"
            if relu:
                output = tf.nn.relu(output, name=scope.name)  # ReLU non-linearity
            return output

    @layer
    def prelu(self, _input, name):
        with tf.variable_scope(name):
            i = int(_input.get_shape()[-1])
            alpha = self.make_var('alpha', shape=(i,))
            output = tf.nn.relu(_input) + tf.multiply(alpha, -tf.nn.relu(-_input))
        return output
        
    @layer
    def relu(self, _input, name):
        return tf.nn.relu(_input,name=name)

    @layer
    def max_pool(self, _input, k_h, k_w, s_h, s_w, name, padding='SAME'):
        """ parameters
            input: input images
            k_h: kernal height
            k_w: kernal width
            s_h: stride height
            s_w: stride width
        """
        self.validate_padding(padding)
        return tf.nn.max_pool(_input,ksize=[1, k_h, k_w, 1],strides=[1, s_h, s_w, 1],padding=padding,name=name)
                              
    @layer
    def avg_pool(self, _input, k_h, k_w, s_h, s_w, name, padding='SAME'):
        """ parameters
            input: input images
            k_h: kernal height
            k_w: kernal width
            s_h: stride height
            s_w: stride width
        """
        self.validate_padding(padding)
        return tf.nn.avg_pool(_input,ksize=[1, k_h, k_w, 1],strides=[1, s_h, s_w, 1],padding=padding,name=name)

    @layer
    def reshape_layer(self, _input, d, name):
        input_shape = tf.shape(_input)
        if name == 'rpn_cls_prob_reshape':
            #
            # transpose: (1, AxH, W, 2) -> (1, 2, AxH, W)
            # reshape: (1, 2xA, H, W)
            # transpose: -> (1, H, W, 2xA)
            return tf.transpose(tf.reshape(tf.transpose(_input, [0, 3, 1, 2]),
                                           [input_shape[0],
                                            int(d),
                                            tf.cast(tf.cast(input_shape[1], tf.float32) / tf.cast(d, tf.float32) * \
                                                    tf.cast(input_shape[3], tf.float32), tf.int32),
                                            input_shape[2]]),
                                [0, 2, 3, 1], name=name)
        else:
            return tf.transpose(tf.reshape(tf.transpose(_input, [0, 3, 1, 2]),
                                           [input_shape[0],
                                            int(d),
                                            tf.cast(tf.cast(input_shape[1], tf.float32) * (tf.cast(input_shape[3], 
                                                    tf.float32) / tf.cast(d, tf.float32)),tf.int32),
                                            input_shape[2]]),
                                [0, 2, 3, 1], name=name)

    @layer
    def spatial_reshape_layer(self, _input, d, name):
        input_shape = tf.shape(_input)
        # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
        return tf.reshape(_input, [input_shape[0],input_shape[1],-1,int(d)])
        
    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,depth_radius=radius,alpha=alpha,beta=beta,bias=bias,name=name)
        
    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)
    
    @layer
    def fc(self, _input, num_out, name, relu=True, weight_decay=0.0005, trainable=True):
        """ parameters
            inp    : input images
            num_out: number of classes
        """
        with tf.variable_scope(name):
            # only use the first input
            if isinstance(_input, tuple):
                _input = _input[0]
                
            input_shape = _input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= int(d)
                feed_in = tf.reshape(_input, [-1, dim])
            else:
                feed_in, dim = (_input, input_shape[-1].value)
                
            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

                
            weights = self.make_var('weights', 
                                    shape=[dim, num_out],
                                    initializer=init_weights, 
                                    trainable=trainable,
                                    regularizer=self.l2_regularizer(weight_decay))
            biases = self.make_var('biases', [num_out], init_biases, trainable)
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=name)
            return fc
            
    @layer
    def lstm_fc(self, _input, d_i, d_o, name, weight_decay=0.0005, trainable=True):
        with tf.variable_scope(name):
            shape = tf.shape(_input)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            _input = tf.reshape(_input, [N * H * W, C])

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', 
                                   [d_i, d_o], 
                                   init_weights, 
                                   trainable,
                                   regularizer=self.l2_regularizer(weight_decay))
            biases = self.make_var('biases', [d_o], init_biases, trainable)

            _O = tf.matmul(_input, kernel) + biases
            return tf.reshape(_O, [N, H, W, int(d_o)])

    @layer
    def softmax_(self, target, axis, name=None):
        """
            Multi dimensional softmax,
            refer to https://github.com/tensorflow/tensorflow/issues/210
            compute softmax along the dimension of target
            the native softmax only supports batch_size x dimension
        """
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target-max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = tf.div(target_exp, normalize, name)
        return softmax
        
    @layer
    def softmax(self, _input, name):
        return tf.nn.softmax(_input, name=name)
            
    @layer
    def spatial_softmax(self, _input, name):
        input_shape = tf.shape(_input) # 1, H, WxA, d
        # d = input.get_shape()[-1]
        return tf.reshape(tf.nn.softmax(tf.reshape(_input, [-1, input_shape[3]])),
                          [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)
                          
    @layer
    def add(self, _input, name):
        """contribution by miraclebiu"""
        return tf.add(_input[0], _input[1])
        
    @layer
    def batch_normalization(self, _input, name, relu=True, is_training=False):
        """contribution by miraclebiu"""
        output = tf.contrib.layers.batch_norm(_input, scale=True, center=True, is_training=is_training, scope=name)
        if relu:
            output = tf.nn.relu(output)
        return output
        
    @layer
    def dropout(self, _input, keep_prob, name):
        return tf.nn.dropout(_input, keep_prob, name=name)
        
    @layer
    def proposal_layer(self, _input, _feat_stride, anchor_scales, cfg_key, name):
        if isinstance(_input[0], tuple):
            input[0] = _input[0][0]
            # input[0] shape is (1, H, W, Ax2)
            # rpn_rois <- (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        with tf.variable_scope(name):
            blob, bbox_delta = tf.py_func(proposal_layer_np,
                                          [_input[0], _input[1], _input[2], cfg_key,_feat_stride, anchor_scales],
                                          [tf.float32, tf.float32])

            rpn_rois = tf.convert_to_tensor(tf.reshape(blob, [-1, 5]), name='rpn_rois')  # shape is (1 x H x W x A, 5)
            rpn_targets = tf.convert_to_tensor(bbox_delta, name='rpn_targets')  # shape is (1 x H x W x A, 4)
            self.layers['rpn_rois'] = rpn_rois
            self.layers['rpn_targets'] = rpn_targets

            return rpn_rois, rpn_targets
            
    @layer
    def anchor_target_layer(self, _input, _feat_stride, anchor_scales, name):
        if isinstance(_input[0], tuple):
            _input[0] = _input[0][0]

        with tf.variable_scope(name):
            # 'rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info'
            rpn_info = tf.py_func(anchor_target_layer_np,
                                  [_input[0], _input[1], _input[2], _input[3],_input[4], _feat_stride, anchor_scales],
                                  [tf.float32, tf.float32, tf.float32, tf.float32])
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_info[0:]

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32),name='rpn_labels')  # shape is (1 x H x W x A, 2)
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets,name='rpn_bbox_targets')  # shape is (1 x H x W x A, 4)
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights,name='rpn_bbox_inside_weights')  # shape is (1 x H x W x A, 4)
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights,name='rpn_bbox_outside_weights')  # shape is (1 x H x W x A, 4)

            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
        
    