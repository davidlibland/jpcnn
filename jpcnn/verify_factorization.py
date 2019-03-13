import tensorflow as tf
import numpy as np
from itertools import product

from jpcnn.dct_utils import basic_compression, flat_compress

tf.enable_eager_execution()
from jpcnn.model import model


image_dim = 8
noise = np.random.beta(1,1,[16, image_dim, image_dim, 1]).astype(dtype=np.float32)
compression = basic_compression(.1, 1., [2, 2])
input = flat_compress(noise, compression)
comp_im_dim = input.shape[-2]
channels = input.shape[-1]

x = tf.constant(input, dtype=tf.float32)
counters = {}
with tf.GradientTape(persistent=True) as g:
  g.watch(x)
  y = model(x, labels=None, avg_num_filters=2, num_layers=1, num_resnet=1,
            compression=compression)[0,:,:,:]
  y_s = [(y[i, j, k], i, j, k)
         for i in range(comp_im_dim)
         for j in range(comp_im_dim)
         for k in range(channels*3)]
dy_dx = g.gradient(y, x) # Will compute to 6.0
dys_dx = [(g.gradient(yijk, x)[0,:,:,:], i, j, k) for yijk, i, j, k in y_s]

for y_ijk, i, j, k in dys_dx:
    for l, m, n in product(range(comp_im_dim), range(comp_im_dim), range(channels)):
        if l > i or (m > j and l == i) or (3*n>=k and m == j and l == i):
            assert np.isclose(y_ijk[l, m, n], 0), \
                "The gradient of the output at (%d, %d, %d) with respect to " \
                "the input at (%d, %d, %d) should be zero" % (i, j, k, l, m, n)

masks = [((np.isclose(dy, 0) == False).astype(np.int), i,j,k) for dy, i, j, k in dys_dx]
for dy, i, j, k in masks:
    print(i,j,k)
    print(dy[i,j,:])
    print(tf.reduce_sum(dy, axis=2))
