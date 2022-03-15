import numpy as np

from ctypes import *
from instance_norm_tf import instance_norm_tf
from instance_norm_np import instance_norm_np, instance_norm_grad_np

lib = cdll.LoadLibrary('./libin.so')

def check_close(ref, x, msg, rtol, atol):
  assert ref.shape == x.shape
  input_shape = ref.shape
  print(f"Checking {msg}...", end='')
 # print(x)
 # print(ref)
 # print('-------------------------------------------------------')
  if not np.allclose(ref, x, rtol=rtol, atol=atol):
    ind = np.argmin(np.isclose(ref, x, rtol=rtol, atol=atol))
    ind = np.unravel_index(ind, input_shape)
    print(f"\nError at {ind}: ref={ref[ind]}, cpp={x[ind]}")
  else:
    print("Pass")


def evaluate_cpp(input_shape, rtol=1e-3, atol=1e-3):
  print(f"Evaluating {input_shape}...")
  assert len(input_shape) == 3
  epsilon = 0.001
  dtype = np.float32

  np.random.seed(12)
  x = np.random.normal(size=input_shape).astype(dtype)
  gamma = np.random.normal(size=input_shape[1]).astype(dtype)
  beta = np.random.normal(size=input_shape[1]).astype(dtype)
 # dy = np.ones(shape=input_shape, dtype=dtype)
  dy = np.random.random(input_shape).astype(dtype)

  #y, dgamma, dbeta, dx = instance_norm_tf(x, gamma, beta, epsilon)

  y_cpp = np.empty_like(x)
  dx_cpp = np.empty_like(x)
  dgamma_cpp = np.empty_like(gamma)
  dbeta_cpp = np.empty_like(beta)

  gamma_np = gamma.reshape((1,input_shape[1] , 1))
  beta_np = beta.reshape((1,input_shape[1] , 1))
  y_np, cache = instance_norm_np(x, gamma_np,beta_np, epsilon)
  dgamma, dbeta, dx = instance_norm_grad_np(dy, gamma_np, cache)

  lib.instance_norm(
      x.ctypes.data_as(POINTER(c_float)),
      gamma.ctypes.data_as(POINTER(c_float)),
      beta.ctypes.data_as(POINTER(c_float)),
      c_int(input_shape[0]),
      c_int(input_shape[1]),
      c_int(input_shape[2]),
      c_float(epsilon),
      y_cpp.ctypes.data_as(POINTER(c_float)))
  lib.instance_norm_grad(
       dy.ctypes.data_as(POINTER(c_float)),
       x.ctypes.data_as(POINTER(c_float)),
       gamma.ctypes.data_as(POINTER(c_float)),
       c_int(input_shape[0]),
       c_int(input_shape[1]),
       c_int(input_shape[2]),
       c_float(epsilon),
       dx_cpp.ctypes.data_as(POINTER(c_float)),
       dgamma_cpp.ctypes.data_as(POINTER(c_float)),
       dbeta_cpp.ctypes.data_as(POINTER(c_float)))

  check_close(y_np, y_cpp, "y", rtol, atol)
  check_close(dgamma, dgamma_cpp, "dgamma", rtol, atol)
  check_close(dbeta, dbeta_cpp, "dbeta", rtol, atol)
  check_close(dx, dx_cpp, "dx", rtol, atol)


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
    if input_shape == (10, 10000000):
      evaluate_cpp(input_shape, 1e-2, 1e-1)
      continue
    evaluate_cpp(input_shape)

