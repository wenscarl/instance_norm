import tensorflow as tf
import tensorflow_addons as tfa
import time

def instance_norm_tf(x, gamma, beta, epsilon):
  x = tf.convert_to_tensor(x)
  input_shape = x.shape
  assert len(input_shape) == 3

  channel_axis = 1
  layer = tfa.layers.InstanceNormalization(axis=channel_axis, center=True,
                                           scale=True, epsilon=epsilon)
  layer.build(input_shape=input_shape)
  layer.set_weights([gamma, beta])

  def train_step(x):
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = layer(x)
      loss = tf.reduce_sum(y)
    dx, dy, (dgamma, dbeta) = tape.gradient(loss, [x, y, layer.variables])
    return y, dgamma, dbeta, dx, dy

  y, dgamma, dbeta, dx, dy = train_step(x)

  dy_is_one = tf.reduce_all(dy == 1.)
  assert dy_is_one.numpy() == True
  return y, dgamma, dbeta, dx

def benchmark_tf(input_shape):
  assert len(input_shape) == 3
  warmup = 10
  repeat = 10
  channel_axis = 1
  layer = tfa.layers.InstanceNormalization(axis=channel_axis)

  def train_step(x):
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = layer(x)
      loss = tf.reduce_sum(y)
    dx, (dgamma, dbeta) = tape.gradient(loss, [x, layer.variables])
    return dx, dgamma, dbeta

  data = tf.random.normal(input_shape,dtype=tf.dtypes.float16)

  for i in range(warmup):
    dx, dgamma, dbeta = train_step(data)
  _ = tf.reduce_sum(dx).numpy()

  start = time.time()
  for i in range(repeat):
    dx, dgamma, dbeta = train_step(data)
  _ = tf.reduce_sum(dx).numpy()

  result = time.time() - start
  print("Time: {:0.2f} ms".format(1000 * result / repeat))


input_shapes = [
#   (10, 64, 400000),
   #(3, 4, 5),
   # in excel
   (2, 32, 128**3),
   (2, 64, 128**3),
   (4, 32, 128**3),
   (4, 64, 64**3),
   (8, 32, 64**3),
   (8, 64, 64**3),
   (8, 128, 64**3),
   (4, 256, 32**3),
   (8, 256, 32**3),
#   (10, 64, 500000),
#   (100, 64, 50000),
#   (1000, 64, 5000),
#   (10000, 64, 1000),
#   (100000, 64, 100),
#   (1000000, 64, 10),
  #  (10, 100, 100000),
  #  (100, 100, 10000),
  #  (1000, 100, 1000),
  #  (10000, 100, 100),
  #  (100000, 100, 10),
  #  (100, 100000, 10),
  #  (100, 10000, 100),
  #  (100, 1000, 1000),
  #  (100, 100, 10000),
  #  (100, 10, 100000),
  #  (100000, 10, 100),
  #  (10000, 100, 100),
  #  (1000, 1000, 100),
  #  (100, 10000, 100),
  #  (10, 100000, 100),
 ]

if __name__ == "__main__":
  for input_shape in input_shapes:
    benchmark_tf(input_shape)

