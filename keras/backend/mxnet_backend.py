import mxnet as mx
import numpy as np
import inspect

from .common import _FLOATX, _EPSILON, image_dim_ordering, reset_uids
py_all = all

def variable(value, dtype=_FLOATX, name=None):
    '''Instantiates a tensor.

    # Arguments
        value: numpy array, initial value of the tensor.
        dtype: tensor type.
        name: optional name string for the tensor.

    # Returns
        Tensor variable instance.
    '''
    # TODO: No support for Sparse matrix yet, convert to Dense.
    if hasattr(value, 'tocoo'):
       value = value.todense()
    v = mx.ndarray.array(value)
    return v

def placeholder(shape=None, ndim=None, dtype=_FLOATX, sparse=False, name=None):
    '''Instantiates a placeholder.

    # Arguments
        shape: shape of the placeholder
            (integer tuple, may include None entries).
        ndim: number of axes of the tensor.
            At least one of {`shape`, `ndim`} must be specified.
            If both are specified, `shape` is used.
        dtype: placeholder type.
        name: optional name string for the placeholder.

    # Returns
        Placeholder tensor instance.
    '''
    if not shape:
        if ndim:
            shape = tuple([None for _ in range(ndim)])
    x = mx.symbol.Variable(name='A', shape=shape)
    return x


def ndim(x):
    '''Returns the number of axes in a tensor, as an integer.
    '''
    return len(x.shape)

def dtype(x):
    '''Returns the dtype of a tensor, as a string.
    '''
    return x.dtype.name

def eval(x):
    '''Evaluates the value of a tensor.
    Returns a Numpy array.
    '''
    if x.shape == (1,):
        return x.asscalar()
    else :
        return x.asnumpy()

# LINEAR ALGEBRA

def dot(x, y):
    return mx.ndarray.dot(x, y);

def batch_dot(x, y, axes):
    return mx.ndarray.batch_dot(x, y);

def transpose(x):
    return mx.ndarray.transpose(x);

def reverse(x, axes):
    '''Reverse a tensor along the the specified axes
    '''
    return mx.ndarray.flip(x, None, axis=axes)

# ELEMENT-WISE OPERATIONS

'''
TODO: There seems to be a bug in MXNET when the functions are called with "None" axis values.
Workaorund with an explicit check.
'''
def max(x, axis=None, keepdims=False):
    '''Maximum value in a tensor.
    '''
    if(axis==None) :
        return mx.ndarray.max(x, keepdims=keepdims);
    else :
        return mx.ndarray.max(x, axis=axis, keepdims=keepdims)

def min(x, axis=None, keepdims=False):
    '''Minimum value in a tensor.
    '''
    if(axis==None) :
        return mx.ndarray.min(x, keepdims=keepdims);
    else :
        return mx.ndarray.min(x, axis=axis, keepdims=keepdims)

def sum(x, axis=None, keepdims=False):
    '''Sum of the values in a tensor, alongside the specified axis.
    '''
    return mx.ndarray.sum(x, axis=axis, keepdims=keepdims)

def argmax(x, axis=None, keepdims=False):
    '''Minimum value in a tensor.
    '''
    if(axis==None) :
        return mx.ndarray.argmax(x, keepdims=keepdims);
    else :
        return mx.ndarray.argmax(x, axis=axis, keepdims=keepdims)

def argmin(x, axis=None, keepdims=False):
    '''Minimum value in a tensor.
    '''
    if(axis==None) :
        return mx.ndarray.argmin(x, keepdims=keepdims);
    else :
        return mx.ndarray.argmin(x, axis=axis, keepdims=keepdims)

def square(x):
    '''Element-wise square .
    '''
    return mx.ndarray.square(x)

def abs(x):
    '''Element-wise absolute value.
    '''
    return mx.ndarray.abs(x)


def sqrt(x):
    '''Element-wise square root.
    '''
    return mx.ndarray.sqrt(x)


def exp(x):
    '''Element-wise exponential.
    '''
    return mx.ndarray.exp(x)


def log(x):
    '''Element-wise log.
    '''
    return mx.ndarray.log(x)


def round(x):
    '''Element-wise rounding to the closest integer.
    '''
    return mx.ndarray.round(x)


def sign(x):
    '''Element-wise sign.
    '''
    return mx.ndarray.sign(x)


def pow(x, a):
    '''Element-wise exponentiation.
    '''
    return mx.ndarray.power(x, a)

def equal(x, y):
    '''Element-wise equality between two tensors.
    Returns a bool tensor.
    '''
    return x == y


def not_equal(x, y):
    '''Element-wise inequality between two tensors.
    Returns a bool tensor.
    '''
    return x != y


def greater(x, y):
    '''Element-wise truth value of (x > y).
    Returns a bool tensor.
    '''
    return x > y


def greater_equal(x, y):
    '''Element-wise truth value of (x >= y).
    Returns a bool tensor.
    '''
    return x >= y


def lesser(x, y):
    '''Element-wise truth value of (x < y).
    Returns a bool tensor.
    '''
    return x < y


def lesser_equal(x, y):
    '''Element-wise truth value of (x <= y).
    Returns a bool tensor.
    '''
    return x <= y


def maximum(x, y):
    '''Element-wise maximum of two tensors.
    '''
    return mx.ndarray.maximum(x, y)


def minimum(x, y):
    '''Element-wise minimum of two tensors.
    '''
    return mx.ndarray.minimum(x, y)


def sin(x):
    '''Computes sin of x element-wise.
    '''
    return mx.ndarray.sin(x)


def cos(x):
    '''Computes cos of x element-wise.
    '''
    return mx.ndarray.cos(x)

# SHAPE OPERATIONS

def concatenate(tensors, axis=-1):
    '''Concantes a list of tensors alongside the specified axis.
    '''
    if axis < 0:
        dims = ndim(tensors[0])
        if dims:
            axis = axis % dims
        else:
            axis = 0
    return mx.ndarray.concatenate(tensors,axis)

def reshape(x, shape):
    '''Reshapes a tensor to the specified shape.
    '''
    return x.reshape(shape)

def permute_dimensions(x, pattern):
    '''Permutes axes in a tensor.

    # Arguments
        pattern: should be a tuple of
            dimension indices, e.g. (0, 2, 1).
    '''
    return x.transpose(x, axes=pattern)

def expand_dims(x, dim=-1):
    '''Adds a 1-sized dimension at index "dim".
    '''
    if dim < 0:
        if ndim(x) == 0:
            dim = 0
        else:
            dim = dim % ndim(x) + 1

    return mx.ndarray.expand_dims(x, axis=dim)

# VALUE MANIPULATION


def get_value(x):
    '''Returns the value of a tensor variable,
    as a Numpy array.
    '''
    return eval(x)

def set_value(x, value):
    '''Sets the value of a tensor variable,
    from a Numpy array.
    '''
    y = mx.ndarray.array(value);
    y.copyto(x)

def count_params(x):
    '''Returns the number of scalars in a tensor.
    '''
    shape = x.shape
    return np.prod([shape[i] for i in range(len(shape))])

def get_variable_shape(x):
    return x.shape

def print_tensor(x, message=''):
    '''Print the message and the tensor when evaluated and return the same
    tensor.
    '''
    print(message)
    return x

# NN OPERATIONS

def relu(x, alpha=0., max_value=None):
    '''Rectified linear unit

    # Arguments
        alpha: slope of negative section.
        max_value: saturation threshold.
    '''
    A = mx.symbol.Variable('A')
    C = mx.symbol.LeakyReLU(data=A, act_type='leaky', slope=alpha, upper_bound=max_value)
    executor = C.bind(ctx=mx.current_context(), args={'A':x})
    executor.forward()
    return executor.outputs[0]

def elu(x, alpha=0.):
    '''Rectified linear unit

    # Arguments
        alpha: slope of negative section.
        max_value: saturation threshold.
    '''
    A = mx.symbol.Variable('A')
    C = mx.symbol.LeakyReLU(data=A, act_type='elu', slope=alpha)
    executor = C.bind(ctx=mx.current_context(), args={'A':x})
    executor.forward()
    return executor.outputs[0]

def tanh(x, alpha=0., max_value=None):
    '''Rectified linear unit

    # Arguments
        alpha: slope of negative section.
        max_value: saturation threshold.
    '''
    A = mx.symbol.Variable('A')
    tanh1 = mx.symbol.Activation(data=A, act_type="tanh")
    executor = tanh1.bind(ctx=mx.current_context(), args={'A':x})
    executor.forward()
    return executor.outputs[0]

def sigmoid(x, alpha=0., max_value=None):
    '''Rectified linear unit

    # Arguments
        alpha: slope of negative section.
        max_value: saturation threshold.
    '''
    A = mx.symbol.Variable('A')
    sigmoid1 = mx.symbol.Activation(data=A, act_type="sigmoid")
    executor = sigmoid1.bind(ctx=mx.current_context(), args={'A':x})
    executor.forward()
    return executor.outputs[0]

def softmax(x):
    '''Softmax of a tensor.
    '''
    A = mx.symbol.Variable('A')
    C = mx.symbol.SoftmaxActivation(data=A)
    executor = C.bind(ctx=mx.current_context(), args={'A':x})
    executor.forward()
    return executor.outputs[0]


def softplus(x):
    '''Softplus of a tensor.
    '''
    A = mx.symbol.Variable('A')
    C = mx.symbol.LeakyReLU(data=A, act_type='leaky', slope=alpha, upper_bound=max_value)
    executor = C.bind(ctx=mx.current_context(), args={'A':x})
    executor.forward()
    return executor.outputs[0]

# GRAPH MANIPULATION

class Function(object):

    def __init__(self, inputs, outputs, updates=[], **kwargs):
        self.function = mx.ndarray.function(inputs, outputs, updates=updates,
                                        allow_input_downcast=True,
                                        on_unused_input='ignore',
                                        **kwargs)

    def __call__(self, inputs):
        assert type(inputs) in {list, tuple}
        return self.function(*inputs)


def function(inputs, outputs, updates=[], **kwargs):
    if len(kwargs) > 0:
        function_args = inspect.getargspec(mx.ndarray.function)[0]
        for key in kwargs.keys():
            if key not in function_args:
                msg = "Invalid argument '%s' passed to K.function" % key
                raise ValueError(msg)
    return Function(inputs, outputs, updates=updates, **kwargs)



# RANDOMNESS

def random_normal(shape, mean=0.0, std=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = np.random.randint(1, 10e6)
    mx.random.seed(seed)
    return mx.random.normal(loc=mean, scale=std, shape=shape)

def random_uniform(shape, low=0.0, high=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = np.random.randint(1, 10e6)
    mx.random.seed(seed)
    return mx.random.uniform(low=low, high=high, shape=shape)

def random_binomial(shape, p=0.0, dtype=_FLOATX, seed=None):
    '''
    TODO: has a bug for negative p. Need ternary operator support in MXNET.
    '''
    if seed is None:
        seed = np.random.randint(1, 10e6)
    # A > p ? zeros : ones
    a = mx.random.uniform(low=-10e6, high=10e6, shape=shape)
    a = (mx.ndarray.sign(a - p) + mx.ndarray.ones(shape)) / 2
    return a

# HIGH ORDER FUNCTIONS
