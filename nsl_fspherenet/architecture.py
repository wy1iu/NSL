import tensorflow as tf
import numpy as np
import itertools

class VGG():
    def get_conv_filter(self, shape, reg, stddev, reg_mag=1.0):
        init = tf.random_normal_initializer(stddev=stddev)
        if reg:
            regu = tf.contrib.layers.l2_regularizer(self.wd * reg_mag)
            filt = tf.get_variable('filter', shape, initializer=init, regularizer=regu)
        else:
            filt = tf.get_variable('filter', shape, initializer=init)

        return filt

    def get_named_conv_filter(self, shape, reg, stddev, name):
        init = tf.random_normal_initializer(stddev=stddev)
        if reg:
            regu = tf.contrib.layers.l2_regularizer(self.wd)
            filt = tf.get_variable(name, shape, initializer=init, regularizer=regu)
        else:
            filt = tf.get_variable(name, shape, initializer=init)

        return filt

    def get_bias(self, dim, init_bias, name):
        with tf.variable_scope(name):
            init = tf.constant_initializer(init_bias)
            regu = tf.contrib.layers.l2_regularizer(self.wd)
            bias = tf.get_variable('bias', dim, initializer=init, regularizer=regu)

            return bias

    def batch_norm(self, x, n_out, phase_train):
        """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope('bn'):

            gamma = self.get_bias(n_out, 1.0, 'gamma')
            beta = self.get_bias(n_out, 0.0, 'beta')

            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.999)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            return tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

    def _max_pool(self, bottom, ksize, name):
        return tf.nn.max_pool(bottom, ksize=[1, ksize, ksize, 1], strides=[1, 2, 2, 1],
            padding='SAME', name=name)

    def _get_input_norm(self, bottom, ksize, stride, pad):
        eps = 1e-4
        shape = [ksize, ksize, bottom.get_shape()[3], 1]
        filt = tf.ones(shape)
        input_norm = tf.sqrt(tf.nn.conv2d(bottom*bottom, filt, [1,stride,stride,1], padding=pad)+eps)
        return input_norm

    def _get_filter_norm(self, filt):
        eps = 1e-4
        return tf.sqrt(tf.reduce_sum(filt*filt, [0, 1, 2], keep_dims=True)+eps)

    def _conv_layer(self, bottom, ksize, n_filt, is_training, name, stride=1, 
        bn=True, pad='SAME', reg=True, relu=False, verbose=True, init_scale=1.0, reg_mag=1.0, norm=False):

        with tf.variable_scope(name) as scope:
            n_input = bottom.get_shape().as_list()[3]
            shape = [ksize, ksize, n_input, n_filt]
            if verbose:
                print("shape of filter %s: %s" % (name, str(shape)))

            filt = self.get_conv_filter(shape, reg, reg_mag=1.0, stddev=tf.sqrt(2.0/tf.to_float(ksize*ksize*n_input))*init_scale)
            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding=pad)

            if norm:
                xnorm = self._get_input_norm(bottom, ksize, stride, pad)
                wnorm = self._get_filter_norm(filt)
                conv /= xnorm
                conv /= wnorm

            if bn:
                conv = self.batch_norm(conv, (1,1,1,n_filt), is_training)
                
            if relu:
                return tf.nn.relu(conv)
            else:
                return conv

    def _1dcond_conv_layer(self, bottom, mask, ksize, n_filt, is_training, name, stride=1, 
        bn=True, pad='SAME', reg=True, relu=False, verbose=True):

        assert pad == 'SAME'
        assert ksize == 3

        with tf.variable_scope(name) as scope:
            n_input =bottom.get_shape().as_list()[3]
            shape = [ksize, ksize, n_input, n_filt]
            _, h, w, _ = bottom.get_shape().as_list()
            if verbose:
                print("shape of filter %s: %s" % (name, str(shape)))

            mask_shape = mask.get_shape().as_list()

            filt = self.get_conv_filter(shape, reg, stddev=tf.sqrt(2.0/tf.to_float(ksize*ksize*n_input)))
            filt = tf.expand_dims(filt, -2)
            aux = np.zeros((3,3,1,9,1))
            for i in range(3):
                for j in range(3):
                    aux[i, j, :, 3*i+j, :] = 1
            filt = tf.reshape(filt * aux, (3,3,n_input, n_filt*9))
            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding=pad)
            conv = tf.reshape(conv, (-1, h, w, 9, n_filt))
            mask = tf.expand_dims(mask, -1)
            conv_res = tf.reduce_sum(conv * (mask), axis=-2)

            if bn:
                conv_res = self.batch_norm(conv_res, (1,1,1,n_filt), is_training)
                
            if relu:
                return tf.nn.relu(conv_res)
            else:
                return conv_res

    def _combined_conv(self, bottom, ksize, n_filt, is_training, name, stride=1, 
        bn=True, pad='SAME', reg=True, relu=False, verbose=True):

        with tf.variable_scope(name) as scope:
            mask = self._conv_layer(bottom, 3, 64, is_training, 'm1',
                                    stride, bn=True, relu=True, reg=reg, reg_mag=10.0, init_scale=0.1, norm=True)
            mask = self._conv_layer(mask, 3, 9, is_training, 'm2',
                                    stride, bn=True, relu=False, reg=reg, reg_mag=10.0, init_scale=0.001, norm=True)
            _, h, w, _ = mask.get_shape().as_list()
            mask += tf.reshape(tf.ones((3,3)), (1,1,1,9))
            conv = self._1dcond_conv_layer(bottom, mask, 3, n_filt, is_training, name='conv',
                                    stride=1, bn=True, relu=True, reg=reg)
            return conv

    def _resnet_unit_v1(self, bottom, ksize, n_filt, is_training, name, stride, reg): 

        with tf.variable_scope(name):

            n_input = bottom.get_shape().as_list()[3]
            residual = self._combined_conv(bottom, ksize, n_filt, is_training, 'first',
                                    stride, bn=True, relu=True, reg=reg)
            residual = self._combined_conv(residual, ksize, n_filt, is_training, name='second',
                                    stride=1, bn=True, relu=False, reg=reg)

            if n_input == n_filt:
                shortcut = bottom
            else:
                shortcut = self._conv_layer(bottom, ksize, n_filt, is_training, 'shortcut', stride, bn=True, relu=False, reg=True)
 
            return tf.nn.relu(residual + shortcut)

    # Input should be an rgb image [batch, height, width, 3]
    def build(self, rgb, n_class, is_training):        
        self.wd = 5e-4     
        ksize = 3
        n_layer = 3

        feat = rgb

        n_out = 32
        feat = self._conv_layer(feat, ksize, n_out, is_training, name='root', reg=True, bn=True, relu=True)

        #32X32
        n_out = 32
        for i in range(n_layer):
            feat = self._combined_conv(feat, ksize, n_out, is_training, name='conv1_' + str(i), reg=True, bn=True, relu=True)

        feat = self._max_pool(feat, 2, 'pool1')

        # Pool 1, 16x16
        n_out = 64
        for i in range(n_layer):
            feat = self._combined_conv(feat, ksize, n_out, is_training, name='conv2_'+str(i), reg=True, bn=True, relu=True)

        feat = self._max_pool(feat, 2, 'pool2')

        # Pool 2, 8x8
        n_out = 128
        for i in range(n_layer):
            feat = self._combined_conv(feat, ksize, n_out, is_training, name='conv3_'+str(i), reg=True, bn=True, relu=True)

        feat = self._max_pool(feat, 2, 'pool3')

        self.score = self._conv_layer(feat, 4, n_class, is_training, "score", bn=False, 
            pad='VALID', reg=True, relu=False)
        self.pred = tf.squeeze(tf.argmax(self.score, axis=3))
