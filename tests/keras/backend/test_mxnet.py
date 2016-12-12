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
       # check_single_tensor_operation('transpose', (4, 2))
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
        #check_single_tensor_operation('permute_dimensions', (4, 2, 3),
        #                              pattern=(2, 0, 1))
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
        #check_single_tensor_operation('argmax', (4, 2))
        #check_single_tensor_operation('argmax', (4, 2), axis=1)

        #check_single_tensor_operation('argmin', (4, 2))
        #check_single_tensor_operation('argmin', (4, 2), axis=1)

        check_single_tensor_operation('square', (4, 2))
        check_single_tensor_operation('abs', (4, 2))
        #check_single_tensor_operation('sqrt', (4, 2)) # nan/0 discrepency.
        check_single_tensor_operation('exp', (4, 2))
        check_single_tensor_operation('log', (4, 2))
        check_single_tensor_operation('round', (4, 2))
        check_single_tensor_operation('sign', (4, 2))
        check_single_tensor_operation('pow', (4, 2), a=3)
        #check_single_tensor_operation('clip', (4, 2), min_value=0.4,
        #                              max_value=0.6)

        # two-tensor ops
        check_two_tensor_operation('equal', (4, 2), (4, 2))
        check_two_tensor_operation('not_equal', (4, 2), (4, 2))
        check_two_tensor_operation('greater', (4, 2), (4, 2))
        check_two_tensor_operation('greater_equal', (4, 2), (4, 2))
        check_two_tensor_operation('lesser', (4, 2), (4, 2))
        check_two_tensor_operation('lesser_equal', (4, 2), (4, 2))
        check_two_tensor_operation('maximum', (4, 2), (4, 2))
        check_two_tensor_operation('minimum', (4, 2), (4, 2))

    def test_nn_operations(self):
        check_single_tensor_operation('relu', (4, 2), alpha=0.1, max_value=0.5)
        check_single_tensor_operation('softmax', (4, 10))
       # check_single_tensor_operation('softplus', (4, 10))
        check_single_tensor_operation('elu', (4, 10), alpha=0.5)

        check_single_tensor_operation('sigmoid', (4, 2))
       # check_single_tensor_operation('hard_sigmoid', (4, 2))
        check_single_tensor_operation('tanh', (4, 2))

        '''
        # dropout
        val = np.random.random((100, 100))
        xth = KTH.variable(val)
        xtf = KTF.variable(val)
        zth = KTH.eval(KTH.dropout(xth, level=0.2))
        ztf = KTF.eval(KTF.dropout(xtf, level=0.2))
        assert zth.shape == ztf.shape
        # dropout patterns are different, only check mean
        assert np.abs(zth.mean() - ztf.mean()) < 0.05

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

if __name__ == '__main__':
    pytest.main([__file__])
