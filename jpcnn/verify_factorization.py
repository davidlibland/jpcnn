import tensorflow as tf
import numpy as np
from itertools import product
tf.enable_eager_execution()
from jpcnn.model import model


image_dim = 4
noise = np.random.beta(1,1,[16, image_dim, image_dim, 1])

x = tf.constant(noise, dtype=tf.float64)
counters = {}
with tf.GradientTape(persistent=True) as g:
  g.watch(x)
  y = model(x, 50, 2, 0)[0,:,:,0]
  y_s = [(y[i, j], i, j)
         for i in range(image_dim)
         for j in range(image_dim)]
dy_dx = g.gradient(y, x) # Will compute to 6.0
dys_dx = [(g.gradient(yij, x)[0,:,:,0], i, j) for yij, i, j in y_s]

for y_ij, i, j in dys_dx:
    for k, l in product(range(image_dim), range(image_dim)):
        if k > i or (l >= j and k == i):
            assert np.isclose(y_ij[k, l], 0), \
                "The gradient of the output at (%d, %d) with respect to " \
                "the input at (%d, %d) should be zero" % (k, l, i, j)

# print(dy_dx)
# print(y)
print(dys_dx)