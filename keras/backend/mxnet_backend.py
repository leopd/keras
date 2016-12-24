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
    if isinstance(x, mx.ndarray.NDArray):
        if(axis == None):
            return mx.ndarray.sum(x)
        else:
            return mx.ndarray.sum(x, axis=axis, keepdims=keepdims)
    else :
        return mx.symbol.sum(x)

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
    #if(axis==None) :
    #    return mx.ndarray.argmin(x, keepdims=keepdims);
    #else :
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
    if isinstance(x, mx.ndarray.NDArray):
        return mx.ndarray.exp(x)
    else :
        return mx.symbol.exp(x)


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

def clip(x, min_value, max_value):
    '''Element-wise value clipping.
    '''
    return mx.ndarray.clip(x,min_value, max_value)

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
    return mx.ndarray.transpose(x, axes=pattern)

def flatten(x):
    A = mx.symbol.Variable('A')
    C = mx.symbol.Flatten(A)
    executor = C.bind(ctx=mx.current_context(), args={'A':x})
    executor.forward()
    return transpose(executor.outputs[0])

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

def dropout(x, level):
    '''Sets entries in `x` to zero at random,
    while scaling the entire tensor.

    # Arguments
        x: tensor
        level: fraction of the entries in the tensor
            that will be set to 0.
    '''

    A = mx.symbol.Variable('A')
    C = mx.symbol.Dropout(data=A, p=level)
    executor = C.bind(ctx=mx.current_context(), args={'A':x})
    executor.forward()
    return executor.outputs[0]

def l2_normalize(x):
    '''Normalizes a tensor wrt the L2 norm alongside the specified axis.
    '''
    return mx.ndarray.norm(x)

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



def gradients(loss, variables):
    return loss.grad([variables[0].name])


def stop_gradient(variables):
    '''Returns `variables` but with zero gradient with respect to every other
    variables.
    '''
    return theano.gradient.disconnected_grad(variables)


# CONVOLUTIONS

def _preprocess_conv2d_input(x, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        x = mx.ndarray.transpose(x,(0, 3, 1, 2))
    return x

def _preprocess_conv3d_input(x, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols, slices)
        # TF input shape: (samples, rows, cols, slices, input_depth)
        x = mx.ndarray.transpose(x,(0, 4, 1, 2, 3))
    return x

def _preprocess_conv2d_kernel(kernel, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        kernel = mx.ndarray.transpose(kernel, (3, 2, 0, 1))
    return kernel

def _preprocess_conv3d_kernel(kernel, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH kernel shape: (depth, input_depth, rows, cols, slices)
        # TF kernel shape: (rows, cols, slices, input_depth, depth)
        kernel = mx.ndarray.transpose(kernel,(4, 3, 0, 1, 2))
    return kernel

def _preprocess_border_mode(border_mode):
    if border_mode == 'same':
        padding = 'SAME'
    elif border_mode == 'valid':
        padding = 'VALID'
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))
    return padding

def _postprocess_conv2d_output(x, dim_ordering):
    if dim_ordering == 'tf':
        x = mx.ndarray.transpose(x,(0, 2, 3, 1))
    return x

def _postprocess_conv3d_output(x, dim_ordering):
    if dim_ordering == 'tf':
        x = mx.ndarray.transpose(x,(0, 2, 3, 4, 1))
    return x

def conv2d(x, kernel, strides=(1, 1), border_mode='valid',
           dim_ordering='default',
           image_shape=None, filter_shape=None, filter_dilation=(1, 1)):
    '''2D convolution.

    # Arguments
        kernel: kernel tensor.
        strides: strides tuple.
        border_mode: string, "same" or "valid".
        dim_ordering: "tf" or "th".
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.
    '''
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv2d_input(x, dim_ordering)
    kernel = _preprocess_conv2d_kernel(kernel, dim_ordering)
    padding = _preprocess_border_mode(border_mode)

    data = mx.sym.Variable(name="data")
    shp = (kernel.shape[2], kernel.shape[3])
    if filter_dilation == (1, 1):
        #strides = (1,) + strides + (1,)
        #x = tf.nn.conv2d(x, kernel, strides, padding=padding)
        conv = mx.sym.Convolution(data=data, kernel=shp, no_bias=True, num_filter=kernel.shape[0], stride=strides, name = "conv")
    else:
        assert filter_dilation[0] == filter_dilation[1]
        assert strides == (1, 1), 'Invalid strides for dilated convolution'
        #x = tf.nn.atrous_conv2d(x, kernel, filter_dilation[0], padding=padding)
        conv = mx.sym.Convolution(data=data, kernel=shp, no_bias=True, num_filter=kernel.shape[0], name = "conv")

    executor = conv.bind(ctx=mx.current_context(), args={'data':x, 'conv_weight':kernel})
    executor.forward()
    y = executor.outputs[0]
    return _postprocess_conv2d_output(y, dim_ordering)

def conv3d(x, kernel, strides=(1, 1, 1),
           border_mode='valid', dim_ordering='default',
           volume_shape=None, filter_shape=None):
    '''3D convolution.

    # Arguments
        kernel: kernel tensor.
        strides: strides tuple.
        border_mode: string, "same" or "valid".
        dim_ordering: "tf" or "th".
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.
    '''
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv3d_input(x, dim_ordering)
    kernel = _preprocess_conv3d_kernel(kernel, dim_ordering)
    padding = _preprocess_border_mode(border_mode)

    data = mx.sym.Variable(name="data")
    shp = (kernel.shape[2], kernel.shape[3], kernel.shape[4])
    conv = mx.sym.Convolution(data=data, kernel=shp, no_bias=True, num_filter=kernel.shape[0], stride=strides,
                              name="conv")
    executor = conv.bind(ctx=mx.current_context(), args={'data': x, 'conv_weight': kernel})
    executor.forward()
    y = executor.outputs[0]

    return _postprocess_conv3d_output(y, dim_ordering)

def pool2d(x, pool_size, strides=(1, 1),
           border_mode='valid', dim_ordering='default',
           pool_mode='max'):
    '''2D Pooling.

    # Arguments
        pool_size: tuple of 2 integers.
        strides: tuple of 2 integers.
        border_mode: one of "valid", "same".
        dim_ordering: one of "th", "tf".
        pool_mode: one of "max", "avg".
    '''
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    padding = _preprocess_border_mode(border_mode)
    x = _preprocess_conv2d_input(x, dim_ordering)

    data = mx.sym.Variable(name="data")
    pool = mx.sym.Pooling(data=data, kernel=pool_size, pool_type=pool_mode, stride=strides, name = "pool")

    executor = pool.bind(ctx=mx.current_context(), args={'data':x})
    executor.forward()
    y = executor.outputs[0]

    return _postprocess_conv2d_output(y, dim_ordering)

def pool3d(x, pool_size, strides=(1, 1, 1), border_mode='valid',
           dim_ordering='default', pool_mode='max'):
    '''3D Pooling.

    # Arguments
        pool_size: tuple of 3 integers.
        strides: tuple of 3 integers.
        border_mode: one of "valid", "same".
        dim_ordering: one of "th", "tf".
        pool_mode: one of "max", "avg".
    '''
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    padding = _preprocess_border_mode(border_mode)
    x = _preprocess_conv3d_input(x, dim_ordering)

    data = mx.sym.Variable(name="data")
    pool = mx.sym.Pooling(data=data, kernel=pool_size, pool_type=pool_mode, stride=strides, name = "pool")

    executor = pool.bind(ctx=mx.current_context(), args={'data':x})
    executor.forward()
    y = executor.outputs[0]

    return _postprocess_conv3d_output(y, dim_ordering)

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

def map_fn(fn, elems, name=None):
    '''Map the function fn over the elements elems and return the outputs.

    # Arguments
        fn: Callable that will be called upon each element in elems
        elems: tensor
        name: A string name for the map node in the graph

    # Returns
        Tensor with first dimension equal to the elems and second depending on
        fn
    '''
    y = fn(mx.ndarray.array(elems), axis=1)
    return y;

def foldl(fn, elems, initializer=None, name=None):
    '''Reduce elems using fn to combine them from left to right.

    # Arguments
        fn: Callable that will be called upon each element in elems and an
            accumulator, for instance lambda acc, x: acc + x
        elems: tensor
        initializer: The first value used (elems[0] in case of None)
        name: A string name for the foldl node in the graph

    # Returns
        Same type and shape as initializer
    '''
    arr = mx.ndarray.array(elems)
    shape = arr.shape
    result = arr[0]
    if(initializer != None):
        result = fn(initializer, result)
    for i in range(1, shape[0]):
        result = fn(result, arr[i])
    return result

def foldr(fn, elems, initializer=None, name=None):
    '''Reduce elems using fn to combine them from right to left.

    # Arguments
        fn: Callable that will be called upon each element in elems and an
            accumulator, for instance lambda acc, x: acc + x
        elems: tensor
        initializer: The first value used (elems[-1] in case of None)
        name: A string name for the foldr node in the graph

    # Returns
        Same type and shape as initializer
    '''
    arr = mx.ndarray.array(elems)
    shape = arr.shape
    # TODO: Some bug in MXNET, the following doesn't work.
    # Rigght now it give incorrect result, the for loop has to end at -1(instead of zero)
    #result = arr[shape[0]-1]
    result = arr[0]
    if(initializer != None):
        result = fn(initializer, result)
    for i in range(shape[0]-1, 0, -1):
        result = fn(result, arr[i])
    return result