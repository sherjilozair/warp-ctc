import numpy as np
import numpy.ctypeslib as npct
import ctypes
import ctypes.util

import theano
import theano.tensor as T
from theano.gradient import DisconnectedType

path = ctypes.util.find_library("warpctc")
libwarpctc = npct.load_library(path, "")

libwarpctc.cpu_ctc.restype = None
libwarpctc.cpu_ctc.argtypes = [
        npct.ndpointer(dtype=np.float32, ndim=3),
        npct.ndpointer(dtype=np.float32, ndim=3),
        npct.ndpointer(dtype=np.int32, ndim=1),
        npct.ndpointer(dtype=np.int32, ndim=1),
        npct.ndpointer(dtype=np.int32, ndim=1),
        ctypes.c_int,
        ctypes.c_int,
        npct.ndpointer(dtype=np.float32, ndim=1),
        ctypes.c_int]

def cpu_ctc(acts, act_lens, labels, label_lens):
    """
    acts: 3-d numpy float array, same as c++ bindings
    act_lens: 1-d int array of input length of each example
    labels: list of 1-d int array for each example in minibatch
    label_lens: 1-d int array of label length of each example
    """
    # make sure correct types
    acts = np.array(acts, dtype=np.float32)
    act_lens = np.array(act_lens, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)
    label_lens = np.array(label_lens, dtype=np.int32)

    # C needs sizes
    alphabet_size = acts.shape[2]
    minibatch = acts.shape[1]

    # create return variables
    grads = np.zeros_like(acts, dtype=np.float32)
    cost = np.zeros((minibatch,), dtype=np.float32)

    # compute
    libwarpctc.cpu_ctc(acts, grads, labels, label_lens, act_lens, alphabet_size, minibatch, cost, 1)
    return cost, grads

class CPUCTCGrad(theano.Op):
    # Properties attribute
    __props__ = ()

    def make_node(self, *inputs):
        inputs = map(theano.tensor.as_tensor_variable, inputs)
        # add checks here for types and numdims of all inputs
        return theano.Apply(self, inputs, [inputs[0].type()])

    def perform(self, node, inputs, outputs):
        cost, gradients = cpu_ctc(*inputs)
        outputs[0][0] = gradients

class CPUCTC(theano.Op):
    # Properties attribute
    __props__ = ()

    def make_node(self, *inputs):
        inputs = map(theano.tensor.as_tensor_variable, inputs)
        # add checks here for types and numdims of all inputs
        return theano.Apply(self, inputs, [T.fvector()])

    def perform(self, node, inputs, outputs):
        cost, gradients = cpu_ctc(*inputs)
        outputs[0][0] = cost

    def grad(self, inputs, output_grads):
        gradients = CPUCTCGrad()(*inputs)
        return [gradients, DisconnectedType()(), DisconnectedType()(), DisconnectedType()()]

if __name__ == '__main__':
    acts = np.array([[[0.1, 0.6, 0.1, 0.1, 0.1]],
                     [[0.1, 0.1, 0.6, 0.1, 0.1]]])
    assert acts.shape == (2, 1, 5) # max_len is 2, minibatch is 1, alphabet_size is 5
    labels = np.array([1, 2])
    label_lens = np.array([2])
    act_lens = np.array([2])
    cost, grads = cpu_ctc(acts, act_lens, labels, label_lens)
    print "cost:", cost.sum()
    print "expected cost:", 2.46285844

    def create_theano_func():
        acts = T.ftensor3()
        act_lens = T.ivector()
        labels = T.ivector()
        label_lens = T.ivector()
        costs = CPUCTC()(acts, act_lens, labels, label_lens)
        cost = T.mean(costs)
        grads = T.grad(cost, acts)
        f = theano.function([acts, act_lens, labels, label_lens], cost, allow_input_downcast=True)   
        g = theano.function([acts, act_lens, labels, label_lens], grads, allow_input_downcast=True)
        return f, g
    f, g = create_theano_func()
    print f(acts, act_lens, labels, label_lens)
    print g(acts, act_lens, labels, label_lens)












