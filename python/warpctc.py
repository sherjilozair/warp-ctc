import numpy as np
import numpy.ctypeslib as npct
import ctypes
import ctypes.util

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

def cpu_ctc(acts, labels, sizes):
    """
    acts: 3-d numpy float array, same as c++ bindings
    labels: list of 1-d int array for each example in minibatch
    sizes: 1-d int array of input size of each example
    """
    acts = np.array(acts, dtype=np.float32)
    sizes = np.array(sizes, dtype=np.int32)
    grads = np.zeros_like(acts, dtype=np.float32)
    label_lengths = np.array([label.size for label in labels], dtype=np.int32)
    flat_labels = np.concatenate(labels).astype(np.int32)
    alphabet_size = acts.shape[2]
    minibatch = acts.shape[1]
    cost = np.zeros((minibatch,), dtype=np.float32)
    libwarpctc.cpu_ctc(acts, grads, flat_labels, label_lengths, sizes, alphabet_size, minibatch, cost, 1)
    return cost, grads

if __name__ == '__main__':
    acts = np.array([[[0.1, 0.6, 0.1, 0.1, 0.1]],
                     [[0.1, 0.1, 0.6, 0.1, 0.1]]])
    assert acts.shape == (2, 1, 5) # max_len is 2, minibatch is 1, alphabet_size is 5
    labels = [np.array([1, 2])]
    sizes = np.array([2])
    cost, grads = cpu_ctc(acts, labels, sizes)
    print "cost:", cost.sum()
    print "expected cost:", 2.46285844

