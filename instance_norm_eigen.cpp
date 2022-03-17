#include <time.h>

#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

template <typename T>
void IsClose2DHost(const T* x, const T* y, int N, int C, int D, std::string msg,
                   float atol = 1e-3, float rtol = 1e-3);

template <typename T>
void Print2DHost(const T* x, int N, int C, int D, std::string msg);

template <typename T, typename U>
void InstanceNormCPU(const T* x, const U* gamma, const U* beta, const int N, const int C,
                  const int D, const U epsilon, T* y);

template <typename T, typename U>
void InstanceNormGradCPU(const T* dy, const T* x, const U* gamma, const int N, const int C,
                      const int D, const U epsilon, U* dgamma, U* dbeta, T* dx);

#define DTYPE float

template <typename T>
void InitAlloc(T* x, int size, int init = -1) {
  srand(12);
  for (int i = 0; i < size; i++) {
    if (init != -1) {
      x[i] = init;
    } else {
      x[i] = static_cast<T>(static_cast<float>(rand()) / RAND_MAX);
    }
  }
}

template <typename T>
void SetEigenTensor(T* out, T* in, int size) {
  for (int i = 0; i < size; i++) {
    out[i] = in[i];
  }
}

template <typename T, typename U>
void InstanceNormEigen(const Eigen::Tensor<T, 3, Eigen::RowMajor>& in,
                    const Eigen::Tensor<U, 1, Eigen::RowMajor>& scale,
                    const Eigen::Tensor<U, 1, Eigen::RowMajor>& offset,
                    const int N, const int C, const int D, const U epsilon,
                    Eigen::Tensor<T, 3, Eigen::RowMajor>& y) {
  Eigen::array<int, 1> reduce_dims({2});
  Eigen::DSizes<Eigen::Index, 3> N_by_C_by_one(N, C, 1);
  Eigen::DSizes<Eigen::Index, 3> one_by_C_by_1(1, C, 1);
  Eigen::array<int, 3> bcast_D({1, 1, D});
  Eigen::array<int, 3> bcast_ND({N, 1, D});
  Eigen::Tensor<float, 1, Eigen::RowMajor> mean(N);
  Eigen::Tensor<float, 1, Eigen::RowMajor> variance(N);

  float D_inv = 1.0f / D;
  auto x = in.template cast<U>();
  mean = x.sum(reduce_dims) * D_inv;

  auto x_centered = x - mean.reshape(N_by_C_by_one).broadcast(bcast_D);

  variance = x_centered.square().sum(reduce_dims) * D_inv;

  auto scaling_factor =
      (variance + epsilon).rsqrt().eval().reshape(N_by_C_by_one).broadcast(bcast_D) *
      scale.reshape(one_by_C_by_1).broadcast(bcast_ND);
  auto x_scaled = x_centered * scaling_factor;

  auto x_shifted = (x_scaled + offset.reshape(one_by_C_by_1).broadcast(bcast_ND));

  y = x_shifted.template cast<T>();
}

template <typename T, typename U>
void InstanceNormGradEigen(const Eigen::Tensor<T, 3, Eigen::RowMajor>& dy,
                        const Eigen::Tensor<T, 3, Eigen::RowMajor>& in,
                        const Eigen::Tensor<U, 1, Eigen::RowMajor>& scale,
                        const int N, const int C, const int D, const U epsilon,
                        Eigen::Tensor<U, 1, Eigen::RowMajor>& dscale,
                        Eigen::Tensor<U, 1, Eigen::RowMajor>& doffset,
                        Eigen::Tensor<T, 3, Eigen::RowMajor>& dx) {
  Eigen::array<int, 1> reduce_D({2});
  Eigen::array<int, 2> reduce_ND({0, 2});
  Eigen::DSizes<Eigen::Index, 3> N_by_C_by_one(N, C, 1);
  Eigen::DSizes<Eigen::Index, 3> one_by_C_by_one(1, C, 1);
  Eigen::array<int, 3> bcast_D({1, 1, D});
  Eigen::array<int, 3> bcast_ND({N, 1, D});
  Eigen::Tensor<float, 2, Eigen::RowMajor> mean(N, C);
  Eigen::Tensor<float, 2, Eigen::RowMajor> ivar(N, C);

  float D_inv = 1.0f / D;
  auto x = in.template cast<U>();
  mean = x.sum(reduce_D) * D_inv;

  auto x_centered = (x - mean.reshape(N_by_C_by_one).broadcast(bcast_D)).eval();

  auto variance = x_centered.square().sum(reduce_D) * D_inv;

 // ivar = (variance + epsilon).rsqrt().eval().reshape(N_by_C_by_one).broadcast(bcast_D);

//  dscale = (dy * x_centered * ivar).sum(reduce_ND);
//  doffset = dy.sum(reduce_ND);
//
//  // Compute dl_di: dy * scale * ivar
//  auto dl_di = (dy * scale.reshape(one_by_C_by_one).broadcast(bcast_ND) * ivar).eval();
//  U di_dx = 1.;
//
//  // Compute dl_dvar: (dy * scale * x_centered * -0.5 * ivar^3).sum(reduce_D)
//  auto dl_dvar =
//      ((dl_di * x_centered * (-0.5f) * ivar * ivar).sum(reduce_D)).eval();
//  auto dvar_dx = (2.f * x_centered * D_inv).eval();
//
//  // Compute dl_mean: (-1 * dy * scale * ivar).sum(reduce_D) + (dl_dvar * -2 / D
//  // * x_centered).sum(reduce_D)
//  auto dl_dmean = (-1.f * dl_di).sum(reduce_D).eval() +
//                  (dl_dvar.reshape(N_by_C_by_one).broadcast(bcast_D) * (-2.f) *
//                   D_inv * x_centered)
//                      .sum(reduce_D)
//                      .eval();
//  U dmean_dx = 1.f * D_inv;
//
//  auto out = dl_di * di_dx +
//             dl_dvar.reshape(N_by_C_by_one).broadcast(bcast_D) * dvar_dx +
//             dl_dmean.reshape(N_by_C_by_one).broadcast(bcast_D) * dmean_dx;
// // dx = out.template cast<T>();
}

int main(int argc, char** argv) {
  int N = 1000;
  int C = 64;
  int D = 10000;
  int allow_print = 0;
  if (argc >= 4) {
    N = atoi(argv[1]);
    C = atoi(argv[2]);
    D = atoi(argv[3]);
    allow_print = atoi(argv[4]);
  }

  DTYPE* x_data = new DTYPE[N * C * D];
  float* gamma_data = new float[C];
  float* beta_data = new float[C];
  InitAlloc(x_data, N * C * D);
  InitAlloc(gamma_data, C);
  InitAlloc(beta_data, C);

  const float epsilon = 0.001f;
  Eigen::Tensor<DTYPE, 3, Eigen::RowMajor> x(N, C, D);
  Eigen::Tensor<DTYPE, 3, Eigen::RowMajor> y(N, C, D);
  Eigen::Tensor<float, 1, Eigen::RowMajor> scale(C);
  Eigen::Tensor<float, 1, Eigen::RowMajor> offset(C);
  SetEigenTensor(x.data(), x_data, N * C * D);
  SetEigenTensor(scale.data(), gamma_data, C);
  SetEigenTensor(offset.data(), beta_data, C);
  double time_spent = 0.0;
  clock_t begin = clock();

  InstanceNormEigen(x, scale, offset, N, C, D, epsilon, y);

  clock_t end = clock();
  time_spent += (double)(end - begin) / (CLOCKS_PER_SEC / 1000);
  printf("Eigen time: %f ms\n", time_spent);

  if (allow_print) {
    std::cout << "Eigen y:" << std::endl;
    std::cout << y << std::endl;
  }

  DTYPE* y_data = new DTYPE[N * C * D];

  time_spent = 0.0;
  begin = clock();

  InstanceNormCPU(x_data, gamma_data, beta_data, N, C, D, epsilon, y_data);

  end = clock();
  time_spent += (double)(end - begin) / (CLOCKS_PER_SEC / 1000);
  printf("CPU time: %f ms\n", time_spent);

  if (allow_print) {
    Print2DHost(y_data, N,C, D, "CPU y:");
  }

  IsClose2DHost(y_data, (float*)y.data(), N, C, D, "y");

//  Eigen::Tensor<DTYPE, 3, Eigen::RowMajor> dy(N,C, D);
//  dy.setConstant(1.);
//  Eigen::Tensor<float, 1, Eigen::RowMajor> dscale(C);
//  Eigen::Tensor<float, 1, Eigen::RowMajor> doffset(C);
//  Eigen::Tensor<DTYPE, 3, Eigen::RowMajor> dx(N, C, D);
//  time_spent = 0.0;
//  begin = clock();
//  InstanceNormGradEigen(dy, x, scale, N, C, D, epsilon, dscale, doffset, dx);
//
//  end = clock();
//  time_spent += (double)(end - begin) / (CLOCKS_PER_SEC / 1000);
//  printf("Eigen Grad time: %f ms\n", time_spent);
//  if (allow_print) {
//    std::cout << "Eigen dgamma:" << std::endl;
//    std::cout << dscale << std::endl;
//    std::cout << "Eigen dbeta:" << std::endl;
//    std::cout << doffset << std::endl;
//    std::cout << "Eigen dx:" << std::endl;
//    std::cout << dx << std::endl;
//  }
//
//  float* dgamma_data = new float[C];
//  float* dbeta_data = new float[C];
//  DTYPE* dx_data = new DTYPE[N * C * D];
//  time_spent = 0.0;
//  begin = clock();
//  InstanceNormGradCPU((DTYPE*)dy.data(), x_data, gamma_data, N, C, D, epsilon,
//                   dgamma_data, dbeta_data, dx_data);
//  end = clock();
//  time_spent += (double)(end - begin) / (CLOCKS_PER_SEC / 1000);
//  printf("CPU Grad time: %f ms\n", time_spent);
//  if (allow_print) {
//    Print2DHost(dgamma_data, 1, 1, C, "CPU dgamma:");
//    Print2DHost(dbeta_data, 1, 1, C, "CPU dbeta:");
//    Print2DHost(dx_data, N, C, D, "CPU dx:");
//  }
//
//  // We need larger atol and rtol mainly when N is too large. Computing dgamma
//  // is essentially a reduction over N dimension.
//  IsClose2DHost(dgamma_data, (float*)dscale.data(), 1, 1, C, "dgamma", 1e-2, 1e-2);
//  IsClose2DHost(dbeta_data, (float*)doffset.data(), 1, 1, C, "dbeta");
//  IsClose2DHost(dx_data, (DTYPE*)dx.data(), N, C, D, "dx");

  delete[] x_data;
  delete[] gamma_data;
  delete[] beta_data;
  delete[] y_data;
//  delete[] dgamma_data;
//  delete[] dbeta_data;
//  delete[] dx_data;
}
