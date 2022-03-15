import numpy as np
import time
import tensorflow as tf
from instance_norm_tf import instance_norm_tf

def ref_instance_norm_all_np(input, gamma, beta, gout, eps):
#  input = np.random.random(
#      size=(batch, channel, height, width)).astype(np.float32)
#  # gamma 初始化为1
#  # beta 初始化为0，所以忽略了
#  #gamma = np.ones((1, channel, 1, 1), dtype=np.float32)
#  gamma = np.random.random((1, channel, 1, 1)).astype(np.float32)
#  beta = np.random.random((1, channel, 1, 1)).astype(np.float32)
#  # 随机生成输出梯度
#  gout = np.random.random(
#      size=(batch, channel, height, width))\
#      .astype(np.float32)
  
  # 用numpy计算前向的结果
  input_shape = input.shape
  batch, channel, height, width = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
  mean_np = np.mean(
    input, axis=(2, 3), keepdims=True)
  in_sub_mean = input - mean_np
  var_np = np.mean(
      np.square(in_sub_mean), 
        axis=(2, 3), keepdims=True)
  invar_np = 1.0 / np.sqrt(var_np + eps)
  out_np = in_sub_mean * invar_np * gamma + beta
  
  # 用numpy计算输入梯度
  scale = 1.0 / (height * width)
  # 对应输入梯度公式第三项
  gvar = gout * gamma * in_sub_mean * \
     -0.5 * np.power(var_np + eps, -1.5)
  gvar = np.sum(gvar, axis=(2, 3), 
          keepdims=True)
  
  # 对应输入梯度公式第二项
  gmean = np.sum(
      gout * gamma, 
      axis=(2, 3), keepdims=True)
  gmean *= -invar_np
  tmp = scale * np.sum(-2.0 * in_sub_mean, 
          axis=(2, 3), keepdims=True) 
  gmean += tmp * gvar
  
  # 对应输入梯度公式三项之和
  gin_np = gout * gamma * invar_np \
      + gvar * scale * 2.0 * in_sub_mean \
      + gmean * scale
  return out_np, gin_np

# (N, C, D)
# gamma, beta (1, C, 1)
def instance_norm_np(x, gamma, beta, epsilon):
  assert len(x.shape) == 3
  D_axis = (2, )

  mean = np.mean(x, axis=D_axis, keepdims=True)
  var = np.var(x, axis=D_axis, keepdims=True)

 # print("NP: mean=", mean)
  x_mean = x - mean
  ivar = 1. / np.sqrt(var + epsilon)
 # print("NP: ivar=", ivar)

  x_normalized = x_mean * ivar
  y = x_normalized * gamma + beta

  cache = {}
  cache["ivar"] = ivar
  cache["x_mean"] = x_mean

  return y, cache

def instance_norm_grad_np(dy, gamma, cache):
  N_axis = (0, )
  D_axis = (2, )
  ND_axis = N_axis + D_axis

  D = 1
  for dim in D_axis:
    D *= dy.shape[dim]

  ivar = cache["ivar"]
  x_mean = cache["x_mean"]
 # print("ivar from np", ivar)

  dgamma = np.sum(dy * x_mean * ivar, axis=ND_axis)
  dbeta = np.sum(dy, axis=ND_axis)

  dl_di = dy * gamma * ivar
  di_dx = 1.0

  dl_dvar = np.sum(dy * gamma * x_mean * (-0.5) * (ivar**3), axis=D_axis,
                   keepdims=True)
  dvar_dx = 2. * x_mean / D

  dl_dmean = np.sum(-1. * dy * gamma * ivar, axis=D_axis, keepdims=True) + \
             dl_dvar * np.sum( (-2. / D) * x_mean, axis=D_axis, keepdims=True)
  dmean_dx = 1. / D

  dx = dl_di * di_dx + dl_dvar * dvar_dx + dl_dmean * dmean_dx
  return dgamma, dbeta, dx


def check_close(ref, x, msg):
  assert ref.shape == x.shape
  input_shape = ref.shape
  print(f"Checking {msg}...", end='')
#  print(ref)
#  print( x)
#  print('---------------------------')
  if not np.allclose(ref, x, rtol=1e-3, atol=1e-3):
    ind = np.argmin(np.isclose(ref, x, rtol=1e-3, atol=1e-3))
    ind = np.unravel_index(ind, input_shape)
    print(f"\nError at {ind}: ref={ref[ind]}, np={x[ind]}")
  else:
    print("Pass")

def evaluate_np(input_shape):
  print(f"Evaluating {input_shape}...")
  assert len(input_shape) == 3
  epsilon = 0.001
  dtype = np.float32

  np.random.seed(12)
  channel_axis = 1
  x = np.random.normal(size=input_shape).astype(dtype)
  gamma = np.random.normal(size=input_shape[channel_axis]).astype(dtype)
  beta = np.random.normal(size=input_shape[channel_axis]).astype(dtype)
  shape_for_np = (1, input_shape[channel_axis], 1)
  gamma_np = gamma.reshape(shape_for_np)
  beta_np = beta.reshape(shape_for_np)

  #dy = np.ones(shape=input_shape, dtype=dtype)
  dy = np.random.random(input_shape).astype(dtype) #(shape=input_shape, dtype=dtype)

  start = time.time()
 # y, dgamma, dbeta, dx = instance_norm_tf(x, gamma, beta, epsilon)
  y_ref, dx_ref = ref_instance_norm_all_np(np.expand_dims(x,axis=-1),
                                           np.expand_dims(gamma_np, axis=-1),
                                           np.expand_dims(beta_np,axis=-1),
                                           np.expand_dims(dy, axis=-1), epsilon)
  mid_t = time.time()

  y_np, cache = instance_norm_np(x, gamma_np, beta_np, epsilon)
  dgamma_np, dbeta_np, dx_np = instance_norm_grad_np(
      dy, gamma.reshape(shape_for_np), cache)
  end_t = time.time()
  print("TF Time: {:0.2f} ms VS NP Time: {:0.2f} ms".format(1000 * (mid_t - start), 1000 * (end_t - mid_t)))

  check_close(y_np, np.squeeze(y_ref, axis=-1), "y")
  check_close(dx_np, np.squeeze(dx_ref, axis=-1), "dx")
#  check_close(y, y_np, "y")
#  check_close(dgamma, dgamma_np, "dgamma")
#  check_close(dbeta, dbeta_np, "dbeta")
#  check_close(dx, dx_np, "dx")

input_shapes = [
   (2, 3, 4),
   (10, 100, 100000),
   (100, 100, 10000),
   (1000, 100, 1000),
   (10000, 100, 100),
   (100000, 100, 10),
   (100, 100000, 10),
   (100, 10000, 100),
   (100, 1000, 1000),
   (100, 100, 10000),
   (100, 10, 100000),
   (100000, 10, 100),
   (10000, 100, 100),
   (1000, 1000, 100),
   (100, 10000, 100),
   (10, 100000, 100),
 ]

if __name__ == "__main__":
  for input_shape in input_shapes:
    evaluate_np(input_shape)
