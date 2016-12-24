import sys
import pytest
from numpy.testing import assert_allclose
import numpy as np
import scipy.sparse as sparse

from keras.backend import theano_backend as KTH
from keras.backend import mxnet_backend as KTF
from keras.utils.np_utils import convert_kernel


def check_single_tensor_operation(function_name, input_shape, **kwargs):
    val = np.random.random(input_shape) - 0.5
    xth = KTH.variable(val)
    xtf = KTF.variable(val)

    zth = KTH.eval(getattr(KTH, function_name)(xth, **kwargs))
    ztf = KTF.eval(getattr(KTF, function_name)(xtf, **kwargs))

    assert zth.shape == ztf.shape
    assert_allclose(zth, ztf, atol=1e-05)


def check_two_tensor_operation(function_name, x_input_shape,
                               y_input_shape, **kwargs):
    xval = np.random.random(x_input_shape) - 0.5

    xth = KTH.variable(xval)
    xtf = KTF.variable(xval)

    yval = np.random.random(y_input_shape) - 0.5

    yth = KTH.variable(yval)
    ytf = KTF.variable(yval)

    zth = KTH.eval(getattr(KTH, function_name)(xth, yth, **kwargs))
    ztf = KTF.eval(getattr(KTF, function_name)(xtf, ytf, **kwargs))

    assert zth.shape == ztf.shape
    assert_allclose(zth, ztf, atol=1e-05)


def check_composed_tensor_operations(first_function_name, first_function_args,
                                     second_function_name, second_function_args,
                                     input_shape):
    ''' Creates a random tensor t0 with shape input_shape and compute
                 t1 = first_function_name(t0, **first_function_args)
                 t2 = second_function_name(t1, **second_function_args)
        with both Theano and TensorFlow backends and ensures the answers match.
    '''
    val = np.random.random(input_shape) - 0.5
    xth = KTH.variable(val)
    xtf = KTF.variable(val)

    yth = getattr(KTH, first_function_name)(xth, **first_function_args)
    ytf = getattr(KTF, first_function_name)(xtf, **first_function_args)

    zth = KTH.eval(getattr(KTH, second_function_name)(yth, **second_function_args))
    ztf = KTF.eval(getattr(KTF, second_function_name)(ytf, **second_function_args))

    assert zth.shape == ztf.shape
    assert_allclose(zth, ztf, atol=1e-05)


class TestBackend(object):

    def test_linear_operations(self):
        check_two_tensor_operation('dot', (4, 2), (2, 4))
        #check_two_tensor_operation('dot', (4, 2), (5, 2, 3))

        #check_two_tensor_operation('batch_dot', (4, 2, 3), (4, 5, 3),
        #                           axes=(2, 2))
        check_single_tensor_operation('transpose', (4, 2))
        #check_single_tensor_operation('reverse', (4, 3, 2), axes=1)
        #check_single_tensor_operation('reverse', (4, 3, 2), axes=(1, 2))

    def test_shape_operations(self):
        # concatenate
        xval = np.random.random((4, 3))
        xth = KTH.variable(xval)
        xtf = KTF.variable(xval)
        yval = np.random.random((4, 2))
        yth = KTH.variable(yval)
        ytf = KTF.variable(yval)
        zth = KTH.eval(KTH.concatenate([xth, yth], axis=-1))
        ztf = KTF.eval(KTF.concatenate([xtf, ytf], axis=-1))
        assert zth.shape == ztf.shape
        assert_allclose(zth, ztf, atol=1e-05)

        check_single_tensor_operation('reshape', (4, 2), shape=(8, 1))
        check_single_tensor_operation('permute_dimensions', (4, 2, 3),
                                      pattern=(2, 0, 1))
        #check_single_tensor_operation('repeat', (4, 1), n=3)
        #check_single_tensor_operation('flatten', (4, 1))
        check_single_tensor_operation('expand_dims', (4, 3), dim=-1)
        check_single_tensor_operation('expand_dims', (4, 3, 2), dim=1)
        #check_single_tensor_operation('squeeze', (4, 3, 1), axis=2)
        #check_single_tensor_operation('squeeze', (4, 1, 1), axis=1)
        #check_composed_tensor_operations('reshape', {'shape': (4, 3, 1, 1)},
        #                                 'squeeze', {'axis': 2},
        #                                 (4, 3, 1, 1))

    def test_value_manipulation(self):
        val = np.random.random((4, 2))
        xth = KTH.variable(val)
        xtf = KTF.variable(val)

        # get_value
        valth = KTH.get_value(xth)
        valtf = KTF.get_value(xtf)
        assert valtf.shape == valth.shape
        assert_allclose(valth, valtf, atol=1e-05)

        # set_value
        val = np.random.random((4, 2))
        KTH.set_value(xth, val)
        KTF.set_value(xtf, val)

        valth = KTH.get_value(xth)
        valtf = KTF.get_value(xtf)
        assert valtf.shape == valth.shape
        assert_allclose(valth, valtf, atol=1e-05)

        # count_params
        assert KTH.count_params(xth) == KTF.count_params(xtf)

        # print_tensor
        #check_single_tensor_operation('print_tensor', ())
        check_single_tensor_operation('print_tensor', (2,))
        check_single_tensor_operation('print_tensor', (4, 3))
        check_single_tensor_operation('print_tensor', (1, 2, 3))

        val = np.random.random((3, 2))
        xth = KTH.variable(val)
        xtf = KTF.variable(val)
        assert KTH.get_variable_shape(xth) == KTF.get_variable_shape(xtf)

    def test_elementwise_operations(self):
        check_single_tensor_operation('max', (4, 2))
        check_single_tensor_operation('max', (4, 2), axis=1, keepdims=True)

        check_single_tensor_operation('min', (4, 2))
        check_single_tensor_operation('min', (4, 2), axis=1, keepdims=True)
        #check_single_tensor_operation('min', (4, 2, 3), axis=[1, -1])
        '''
        check_single_tensor_operation('mean', (4, 2))
        check_single_tensor_operation('mean', (4, 2), axis=1, keepdims=True)
        check_single_tensor_operation('mean', (4, 2, 3), axis=-1, keepdims=True)
        check_single_tensor_operation('mean', (4, 2, 3), axis=[1, -1])

        check_single_tensor_operation('std', (4, 2))
        check_single_tensor_operation('std', (4, 2), axis=1, keepdims=True)
        check_single_tensor_operation('std', (4, 2, 3), axis=[1, -1])

        check_single_tensor_operation('prod', (4, 2))
        check_single_tensor_operation('prod', (4, 2), axis=1, keepdims=True)
        check_single_tensor_operation('prod', (4, 2, 3), axis=[1, -1])

        # does not work yet, wait for bool <-> int casting in TF (coming soon)
        # check_single_tensor_operation('any', (4, 2))
        # check_single_tensor_operation('any', (4, 2), axis=1, keepdims=True)
        #
        # check_single_tensor_operation('any', (4, 2))
        # check_single_tensor_operation('any', (4, 2), axis=1, keepdims=True)
'''
        #check_single_tensor_operation('argmax', (4, 2)) #Global reduction not yet supported
        check_single_tensor_operation('argmax', (4, 2), axis=1)

        #check_single_tensor_operation('argmin', (4, 2)) #Global reduction not yet supported
        check_single_tensor_operation('argmin', (4, 2), axis=1)

        check_single_tensor_operation('square', (4, 2))
        check_single_tensor_operation('abs', (4, 2))
        #check_single_tensor_operation('sqrt', (4, 2)) # nan/0 discrepency.
        check_single_tensor_operation('exp', (4, 2))
        check_single_tensor_operation('log', (4, 2))
        check_single_tensor_operation('round', (4, 2))
        check_single_tensor_operation('sign', (4, 2))
        check_single_tensor_operation('pow', (4, 2), a=3)
        check_single_tensor_operation('clip', (4, 2), min_value=0.4, max_value=0.6)

        # two-tensor ops
        check_two_tensor_operation('equal', (4, 2), (4, 2))
        check_two_tensor_operation('not_equal', (4, 2), (4, 2))
        check_two_tensor_operation('greater', (4, 2), (4, 2))
        check_two_tensor_operation('greater_equal', (4, 2), (4, 2))
        check_two_tensor_operation('lesser', (4, 2), (4, 2))
        check_two_tensor_operation('lesser_equal', (4, 2), (4, 2))
        check_two_tensor_operation('maximum', (4, 2), (4, 2))
        check_two_tensor_operation('minimum', (4, 2), (4, 2))

    def test_conv2d(self):
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)

        for input_shape in [(2, 3, 4, 5), (2, 3, 5, 6)]:
            for kernel_shape in [(4, 3, 2, 2), (4, 3, 3, 4)]:
                xval = np.random.random(input_shape)

                xth = KTH.variable(xval)
                xtf = KTF.variable(xval)

                kernel_val = np.random.random(kernel_shape) - 0.5

                kernel_th = KTH.variable(convert_kernel(kernel_val, dim_ordering='th'))
                kernel_tf = KTF.variable(kernel_val)

                zth = KTH.eval(KTH.conv2d(xth, kernel_th, dim_ordering='th'))
                ztf = KTF.eval(KTF.conv2d(xtf, kernel_tf, dim_ordering='th'))

                assert zth.shape == ztf.shape
                assert_allclose(zth, ztf, atol=1e-05)

        input_shape = (1, 6, 5, 3)
        kernel_shape = (3, 3, 3, 2)

        xval = np.random.random(input_shape)

        xth = KTH.variable(xval)
        xtf = KTF.variable(xval)

        kernel_val = np.random.random(kernel_shape) - 0.5

        kernel_th = KTH.variable(convert_kernel(kernel_val, dim_ordering='tf'))
        kernel_tf = KTF.variable(kernel_val)

        zth = KTH.eval(KTH.conv2d(xth, kernel_th, dim_ordering='tf'))
        ztf = KTF.eval(KTF.conv2d(xtf, kernel_tf, dim_ordering='tf'))

        assert zth.shape == ztf.shape
        assert_allclose(zth, ztf, atol=1e-05)

    @pytest.mark.skip(reason="MXNET missing support for 3D Convolution: https://github.com/dmlc/mxnet/issues/4301 ")
    def test_conv3d(self):
        # TH input shape: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3)
        # TF input shape: (samples, conv_dim1, conv_dim2, conv_dim3, input_depth)
        # TH kernel shape: (depth, input_depth, x, y, z)
        # TF kernel shape: (x, y, z, input_depth, depth)

        # test in dim_ordering = th
        for input_shape in [(2, 3, 4, 5, 4), (2, 3, 5, 4, 6)]:
            for kernel_shape in [(4, 3, 2, 2, 2), (4, 3, 3, 2, 4)]:
                xval = np.random.random(input_shape)

                xth = KTH.variable(xval)
                xtf = KTF.variable(xval)

                kernel_val = np.random.random(kernel_shape) - 0.5

                kernel_th = KTH.variable(convert_kernel(kernel_val, dim_ordering='th'))
                kernel_tf = KTF.variable(kernel_val)

                zth = KTH.eval(KTH.conv3d(xth, kernel_th, dim_ordering='th'))
                ztf = KTF.eval(KTF.conv3d(xtf, kernel_tf, dim_ordering='th'))

                assert zth.shape == ztf.shape
                assert_allclose(zth, ztf, atol=1e-05)

        # test in dim_ordering = tf
        input_shape = (1, 2, 2, 2, 1)
        kernel_shape = (2, 2, 2, 1, 1)

        xval = np.random.random(input_shape)

        xth = KTH.variable(xval)
        xtf = KTF.variable(xval)

        kernel_val = np.random.random(kernel_shape) - 0.5

        kernel_th = KTH.variable(convert_kernel(kernel_val, dim_ordering='tf'))
        kernel_tf = KTF.variable(kernel_val)

        zth = KTH.eval(KTH.conv3d(xth, kernel_th, dim_ordering='tf'))
        ztf = KTF.eval(KTF.conv3d(xtf, kernel_tf, dim_ordering='tf'))

        assert zth.shape == ztf.shape
        assert_allclose(zth, ztf, atol=1e-05)

    def test_pool2d(self):
        check_single_tensor_operation('pool2d', (5, 10, 12, 3), pool_size=(2, 2),
                                      strides=(1, 1), border_mode='valid')

        check_single_tensor_operation('pool2d', (5, 9, 11, 3), pool_size=(2, 2),
                                      strides=(1, 1), border_mode='valid')

        check_single_tensor_operation('pool2d', (5, 9, 11, 3), pool_size=(2, 3),
                                      strides=(1, 1), border_mode='valid')

    @pytest.mark.skip(reason="MXNET missing support for 3D Pooling. '3D kernel not implemented' error ")
    def test_pool3d(self):
        check_single_tensor_operation('pool3d', (5, 10, 12, 5, 3), pool_size=(2, 2, 2),
                                      strides=(1, 1, 1), border_mode='valid')

        check_single_tensor_operation('pool3d', (5, 9, 11, 5, 3), pool_size=(2, 2, 2),
                                      strides=(1, 1, 1), border_mode='valid')

        check_single_tensor_operation('pool3d', (5, 9, 11, 5, 3), pool_size=(2, 3, 2),
                                      strides=(1, 1, 1), border_mode='valid')


    def test_nn_operations(self):
        check_single_tensor_operation('relu', (4, 2), alpha=0.1, max_value=0.5)
        check_single_tensor_operation('softmax', (4, 10))
       # check_single_tensor_operation('softplus', (4, 10))
        check_single_tensor_operation('elu', (4, 10), alpha=0.5)

        check_single_tensor_operation('sigmoid', (4, 2))
       # check_single_tensor_operation('hard_sigmoid', (4, 2))
        check_single_tensor_operation('tanh', (4, 2))

        # dropout
        val = np.random.random((100, 100))
        xth = KTH.variable(val)
        xtf = KTF.variable(val)
        zth = KTH.eval(KTH.dropout(xth, level=0.2))
        ztf = KTF.eval(KTF.dropout(xtf, level=0.2))
        assert zth.shape == ztf.shape
        # dropout patterns are different, only check mean
        assert np.abs(zth.mean() - ztf.mean()) < 0.05

        '''
        check_two_tensor_operation('binary_crossentropy', (4, 2), (4, 2), from_logits=True)
        check_two_tensor_operation('categorical_crossentropy', (4, 2), (4, 2), from_logits=True)
        check_two_tensor_operation('binary_crossentropy', (4, 2), (4, 2), from_logits=False)
        check_two_tensor_operation('categorical_crossentropy', (4, 2), (4, 2), from_logits=False)

        check_single_tensor_operation('l2_normalize', (4, 3), axis=-1)
        check_single_tensor_operation('l2_normalize', (4, 3), axis=1)
        '''

    def test_random_normal(self):
        mean = 0.
        std = 1.
        rand = KTF.eval(KTF.random_normal((1000, 1000), mean=mean, std=std))
        assert(rand.shape == (1000, 1000))
        assert(np.abs(np.mean(rand) - mean) < 0.01)
        assert(np.abs(np.std(rand) - std) < 0.01)

        rand = KTH.eval(KTH.random_normal((1000, 1000), mean=mean, std=std))
        assert(rand.shape == (1000, 1000))
        assert(np.abs(np.mean(rand) - mean) < 0.01)
        assert(np.abs(np.std(rand) - std) < 0.01)

    def test_random_uniform(self):
        min = -1.
        max = 1.
        rand = KTF.eval(KTF.random_uniform((1000, 1000), min, max))
        assert(rand.shape == (1000, 1000))
        assert(np.abs(np.mean(rand)) < 0.01)
        assert(np.max(rand) <= max)
        assert(np.min(rand) >= min)

        rand = KTH.eval(KTH.random_uniform((1000, 1000), min, max))
        assert(rand.shape == (1000, 1000))
        assert(np.abs(np.mean(rand)) < 0.01)
        assert(np.max(rand) <= max)
        assert(np.min(rand) >= min)

    def test_random_binomial(self):
        p = 0.5
        rand = KTF.eval(KTF.random_binomial((1000, 1000), p))
        assert(rand.shape == (1000, 1000))
        assert(np.abs(np.mean(rand) - p) < 0.01)
        assert(np.max(rand) == 1)
        assert(np.min(rand) == 0)

        rand = KTH.eval(KTH.random_binomial((1000, 1000), p))
        assert(rand.shape == (1000, 1000))
        assert(np.abs(np.mean(rand) - p) < 0.01)
        assert(np.max(rand) == 1)
        assert(np.min(rand) == 0)

    def test_sparse_dot(self):
        x_d = np.array([0, 7, 2, 3], dtype=np.float32)
        x_r = np.array([0, 2, 2, 3], dtype=np.int64)
        x_c = np.array([4, 3, 2, 3], dtype=np.int64)

        x_sparse = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))
        x_dense = x_sparse.toarray()

        W = np.random.random((5, 4))

        backends = [KTF]
        if KTH.th_sparse_module:
            # Theano has some dependency issues for sparse
            backends.append(KTH)

        for K in backends:
            t_W = K.variable(W)
            k_s = K.eval(K.dot(K.variable(x_sparse), t_W))
            k_d = K.eval(K.dot(K.variable(x_dense), t_W))

            assert k_s.shape == k_d.shape
            assert_allclose(k_s, k_d, atol=1e-05)

    def test_sparse_concat(self):
        x_d = np.array([0, 7, 2, 3], dtype=np.float32)
        x_r = np.array([0, 2, 2, 3], dtype=np.int64)
        x_c = np.array([4, 3, 2, 3], dtype=np.int64)

        x_sparse_1 = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))

        x_d = np.array([0, 7, 2, 3], dtype=np.float32)
        x_r = np.array([0, 2, 2, 3], dtype=np.int64)
        x_c = np.array([4, 3, 2, 3], dtype=np.int64)

        x_sparse_2 = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))

        x_dense_1 = x_sparse_1.toarray()
        x_dense_2 = x_sparse_2.toarray()

        backends = [KTF]
        if KTH.th_sparse_module:
            # Theano has some dependency issues for sparse
            backends.append(KTH)

        for K in backends:
            k_s = K.concatenate([K.variable(x_sparse_1), K.variable(x_sparse_2)])
            #assert K.is_sparse(k_s) #TODO enable after sparse support

            k_s_d = K.eval(k_s)

            k_d = K.eval(K.concatenate([K.variable(x_dense_1), K.variable(x_dense_2)]))

            assert k_s_d.shape == k_d.shape
            assert_allclose(k_s_d, k_d, atol=1e-05)

    def test_map(self):
        x = np.random.rand(10, 3).astype(np.float32)
        for K in [KTF, KTH]:
            kx = K.eval(K.map_fn(K.sum, x))

            assert (10,) == kx.shape
            assert_allclose(x.sum(axis=1), kx, atol=1e-05)

    def test_foldl(self):
        x = np.random.rand(10, 3).astype(np.float32)
        for K in [KTF, KTH]:
            kx = K.eval(K.foldl(lambda a, b: a+b, x))

            assert (3,) == kx.shape
            assert_allclose(x.sum(axis=0), kx, atol=1e-05)

    def test_foldr(self):
        # This test aims to make sure that we walk the array from right to left
        # and checks it in the following way: multiplying left to right 1e-40
        # cannot be held into a float32 so it causes an underflow while from
        # right to left we have no such problem and the result is larger
        x = np.array([1e-20, 1e-20, 10, 10, 10], dtype=np.float32)
        for K in [KTF, KTH]:
            p1 = K.eval(K.foldl(lambda a, b: a*b, x))
            p2 = K.eval(K.foldr(lambda a, b: a*b, x))

            assert p1 < p2
            assert 9e-38 < p2 <= 1e-37

if __name__ == '__main__':
    pytest.main([__file__])
