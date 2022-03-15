#include <cub/block/block_reduce.cuh>
#include <iostream>

#define checkCUDA(expression)                               \
  {                                                         \
    cudaError_t status = (expression);                      \
    if (status != cudaSuccess) {                            \
      std::cerr << "Error on line " << __LINE__ << ": "     \
                << cudaGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                              \
    }                                                       \
  }

template <typename T>
void IsClose2DHost(const T* x, const T* y, int N, int C, int D, std::string msg,
                   float atol, float rtol);

template <typename T>
void Print2DHost(const T* x, int N, int C, int D, std::string msg);

template <typename T, typename U>
void InstanceNormCPU(const T* x, const U* gamma, const U* beta, const int N, const int C,
                    const int D, const U epsilon, T* y);

template <typename T, typename U>
void InstanceNormGradCPU(const T* dy, const T* x, const U* gamma, const int N, const int C,
                      const int D, const U epsilon, U* dgamma, U* dbeta, T* dx);

template <typename T, typename U>
void InstanceNormCPUHelper(const T* x, const U* gamma, const U* beta, const int N, const int C,
                        const int D, const U epsilon, T* y_h) {
  T* x_h = new T[N * C * D];
  U* gamma_h = new U[C];
  U* beta_h = new U[C];

  checkCUDA(cudaMemcpy(x_h, x, N * C * D * sizeof(T), cudaMemcpyDeviceToHost));
  checkCUDA(cudaMemcpy(gamma_h, gamma, C * sizeof(U), cudaMemcpyDeviceToHost));
  checkCUDA(cudaMemcpy(beta_h, beta, C * sizeof(U), cudaMemcpyDeviceToHost));

  InstanceNormCPU(x_h, gamma_h, beta_h, N, C, D, epsilon, y_h);

  delete[] x_h;
  delete[] gamma_h;
  delete[] beta_h;
}

template <typename T, typename U>
void InstanceNormGradCPUHelper(const T* dy, const T* x, const U* gamma,
                            const int N, const int C, const int D, const U epsilon,
                            U* dgamma_h, U* dbeta_h, T* dx_h) {
  T* dy_h = new T[N * C * D];
  T* x_h = new T[N * C * D];
  U* gamma_h = new U[C];
  checkCUDA(cudaMemcpy(dy_h, dy, N * C * D * sizeof(T), cudaMemcpyDeviceToHost));
  checkCUDA(cudaMemcpy(x_h, x, N * C * D * sizeof(T), cudaMemcpyDeviceToHost));
  checkCUDA(cudaMemcpy(gamma_h, gamma, C * sizeof(U), cudaMemcpyDeviceToHost));

  InstanceNormGradCPU(dy_h, x_h, gamma_h, N, C, D, epsilon, dgamma_h, dbeta_h, dx_h);

  delete[] dy_h;
  delete[] x_h;
  delete[] gamma_h;
}

const int kBlockSize = 128;
const int kWarpSize = 32;

int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void PrepareAlloc(T** x, int size, int init = -1) {
  srand(12);
  T* buf = new T[size];
  for (int i = 0; i < size; i++) {
    if (init != -1) {
      buf[i] = init;
    } else {
      buf[i] = static_cast<T>(static_cast<float>(rand()) / RAND_MAX);
    }
  }

  checkCUDA(cudaMalloc(&(*x), size * sizeof(T)));
  checkCUDA(cudaMemcpy(*x, buf, size * sizeof(T), cudaMemcpyHostToDevice));

  delete[] buf;
}

template <typename T>
void Print2D(const T* x, int N, int C, int D, std::string msg) {
  T* buf = new T[N * D];
  checkCUDA(cudaMemcpy(buf, x, N * C * D * sizeof(T), cudaMemcpyDeviceToHost));
  Print2DHost(buf, N, C, D, msg);
  delete[] buf;
}

template <typename T>
void IsClose2D(const T* x, const T* y, int N, int C, int D, std::string msg,
               float atol = 1e-3, float rtol = 1e-3) {
  if (D == 10000000) {  // Mainly for y when NxD=10x10000000
    atol = 1e-1;
  }
  if (D == 1000000) {  // Mainly for y when NxD=100x1000000
    atol = 1e-2;
  }
  if (D == 10) {  // Mainly for dgamma when NxD=10000000x10
    atol = 1e-1;
  }
  T* buf = new T[N * C * D];
  checkCUDA(cudaMemcpy(buf, x, N * C * D * sizeof(T), cudaMemcpyDeviceToHost));
  IsClose2DHost(buf, y, N, C, D, msg, atol, rtol);
  delete[] buf;
}

template <typename T, typename U>
__host__ __device__ U GetAs(const T* __restrict__ in, int offset) {
  return static_cast<U>(in[offset]);
}
///////////////////////////// MAYBE inspect
template <typename T, typename U>
struct MeanOp {
  int D;
  __device__ U Compute(const T* x, const int& row, const int& col) const {
    return GetAs<T, U>(x, row * D + col);
  }
  __device__ U Finalize(const U& sum) const { return sum / D; }
};

template <typename T, typename U>
struct IvarOp {
  const U* cache_mean;
  int D;
  U epsilon;
  __device__ U Compute(const T* x, const int& row, const int& col,
                       const U& mean) const {
    U curr = GetAs<T, U>(x, row * D + col);
    return (curr - mean) * (curr - mean);
  }
  __device__ U Compute(const T* x, const int& row, const int& col) const {
    return Compute(x, row, col, cache_mean[row]);
  }
  __device__ U Finalize(const U& sum) const { return rsqrt(sum / D + epsilon); }
};

template <typename T, typename U>
struct DvarOp {
  const U* gamma;
  const T* x;
  const U* cache_ivar;
  const U* cache_mean;
  int D;
  __device__ U Compute(const T* dy, const int& row, const int& col) const {
    U curr = GetAs<T, U>(dy, row * D + col);
    return curr * gamma[col] * (x[row * D + col] - cache_mean[row]) * (-0.5) *
           (cache_ivar[row] * cache_ivar[row] * cache_ivar[row]);
  }
  __device__ U Finalize(const U& sum) const { return sum; }
};

template <typename T, typename U>
struct DmeanOp {
  const U* gamma;
  const T* x;
  const U* cache_ivar;
  const U* cache_mean;
  const U* dl_dvars;
  int D;
  __device__ U Compute(const T* dy, const int& row, const int& col,
                       const U& dl_dvar) const {
    U curr = GetAs<T, U>(dy, row * D + col);
    return -1. * curr * gamma[col] * cache_ivar[row] +
           dl_dvar * (-2. / D) * (x[row * D + col] - cache_mean[row]);
  }
  __device__ U Compute(const T* dy, const int& row, const int& col) const {
    return Compute(dy, row, col, dl_dvars[row]);
  }
  __device__ U Finalize(const U& sum) const { return sum; }
};

template <typename T, typename U>
struct DxOp {
  const T* x;
  const U* cache_mean;
  const U* cache_ivar;
  const U* gamma;
  const U* dl_dvars;
  const U* dl_dmus;
  int D;
  __device__ T Compute(const T* dy, const int& row, const int& col) const {
    U curr = GetAs<T, U>(dy, row * D + col);
    U dl_di = curr * gamma[col] * cache_ivar[row];
    U di_dx = 1.;
    U dvar_dx = 2. * (x[row * D + col] - cache_mean[row]) / D;
    U dmu_dx = 1. / D;
    U dl_dx = dl_di * di_dx + dl_dvars[row] * dvar_dx + dl_dmus[row] * dmu_dx;
    return static_cast<T>(dl_dx);
  }
};

template <typename T, typename U>
struct YOp {
  const U* cache_mean;
  const U* cache_ivar;
  const U* gamma;
  const U* beta;
  int D;
  __device__ T Compute(const T* x, const int& row, const int& col) const {
    U mean = cache_mean[row];
    U ivar = cache_ivar[row];
    U curr = GetAs<T, U>(x, row * D + col);
    return static_cast<T>((curr - mean) * ivar * gamma[col] + beta[col]);
  }
};

template <typename T, typename Op>
__global__ void InstanceNormUpdate(const T* __restrict__ in, const int N, const int C,
                                const int D, T* out, Op op) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= N * C * D) return;

  const int col = tid % D;
  const int row = tid / D;
  out[tid] = op.Compute(in, row, col);
}

template <typename T, typename U>
void InstanceNormGPU(const T* x, const U* gamma, const U* beta, const U epsilon,
                  const int N, const int C, const int D, T* y, U* cache_mean,
                  U* cache_ivar) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  bool use_single_warp = (D <= kWarpSize);

  const int min_num_blocks = kWarpSize;
  const int min_workload_per_thread = 100;
  bool use_single_block =
      (D <= min_num_blocks * kBlockSize * min_workload_per_thread);

  MeanOp<T, U> mean_ops{D};
  IvarOp<T, U> ivar_ops{cache_mean, D, epsilon};

  cudaEventRecord(start);
  int NxC = N * C;
  if (use_single_warp) {
    printf("XLOG: Mean/Var -> single-warp per row\n");
    InstanceNormRowReduceInToOutWarp<<<DivUp(NxC, kBlockSize / kWarpSize),
                                    kBlockSize>>>(
        x, N, C, D, cache_mean, cache_ivar, mean_ops, ivar_ops);
  } else if (use_single_block) {
    // printf("XLOG: Mean/Var -> single-block per row\n");
    // LayerNormRowReduceInToOut<<<N, kBlockSize>>>(
    //     x, N, D, cache_mean, cache_ivar, mean_ops, ivar_ops);
  } else {
    // printf("XLOG: Mean/Var -> multi-block per row\n");
    // const int blocks_per_row = DivUp(D, kBlockSize * min_workload_per_thread);

    // float* temp_sum;
    // float* temp_ivar;
    // PrepareAlloc(&temp_sum, N * blocks_per_row);
    // PrepareAlloc(&temp_ivar, N * blocks_per_row);

    // dim3 threads(kBlockSize, 1, 1);
    // dim3 blocks(blocks_per_row, N, 1);
    // printf("XLOG: num_blocks per row=%d\n", blocks.x);

    // // For long rows, we launch n blocks to process each row. The intermediate
    // // results are stored in a temp memory with the size of N*n. Then, we launch
    // // single block to handle each row of the temp memory.
    // LayerNormRowReduceInToTemp<<<blocks, threads>>>(x, N, D, temp_sum,
    //                                                 mean_ops);
    // LayerNormRowReduceTempToOut<<<N, threads>>>(temp_sum, N, blocks_per_row,
    //                                             cache_mean, mean_ops);

    // LayerNormRowReduceInToTemp<<<blocks, threads>>>(x, N, D, temp_ivar,
    //                                                 ivar_ops);
    // LayerNormRowReduceTempToOut<<<N, threads>>>(temp_ivar, N, blocks_per_row,
    //                                             cache_ivar, ivar_ops);

    // checkCUDA(cudaFree(temp_ivar));
    // checkCUDA(cudaFree(temp_sum));
  }
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds_reduce = 0;
  cudaEventElapsedTime(&milliseconds_reduce, start, stop);

  cudaEventRecord(start);
  YOp<T, U> y_ops{cache_mean, cache_ivar, gamma, beta, D};
  InstanceNormUpdate<<<DivUp(N * D, kBlockSize), kBlockSize>>>(x, N, C, D, y, y_ops);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds_update = 0;
  cudaEventElapsedTime(&milliseconds_update, start, stop);
  printf("InstanceNormGPU time %.2f ms (reduce=%f, update=%f)\n",
         milliseconds_reduce + milliseconds_update, milliseconds_reduce,
         milliseconds_update);
}

template <typename T, typename U, typename Op1, typename Op2>
__global__ void InstanceNormRowReduceInToOutWarp(const T* __restrict__ in,
                                              const int N, const int C,
                                              const int D, U* out1,
                                              U* out2, Op1 op1, Op2 op2) {
  // cache_mean, cache_ivar, mean_ops, ivar_ops
  const int tid = threadIdx.x % kWarpSize;

  const int num_warps = kBlockSize / kWarpSize;
  typedef cub::WarpReduce<U> WarpReduce;
  typename WarpReduce::TempStorage temp_storage[num_warps];

  const int local_warp_id = threadIdx.x / kWarpSize;
  const int warp_id = blockIdx.x * num_warps + local_warp_id;
  int NxC = N * C;
  for (int k = warp_id; k < NxC; k += gridDim.x * num_warps) {
    U partial_sum = 0;
    for (int i = tid; i < D; i += kWarpSize) {
      partial_sum += op1.Compute(in, k, i);
    }

    U sum = WarpReduce(temp_storage[local_warp_id]).Sum(partial_sum);

    sum = cub::ShuffleIndex<kWarpSize>(sum, 0, 0xffffffff);
    sum = op1.Finalize(sum);
    if (tid == 0) {
      out1[k] = sum;
    }

    partial_sum = 0;
    for (int i = tid; i < D; i += kWarpSize) {
      partial_sum += op2.Compute(in, k, i, sum);
    }

    sum = WarpReduce(temp_storage[local_warp_id]).Sum(partial_sum);

    if (tid == 0) {
      out2[k] = op2.Finalize(sum);
    }
  }
}

#define DTYPE float

int main(int argc, char** argv) {
  /** Parameters and Knobs **/
  int N = 2;
  int C = 3;
  int D = 4;
  if (argc >= 3) {
    N = atoi(argv[1]);
    C = atoi(argv[2]);
    D = atoi(argv[3]);
  }
  bool allow_print = false;

  DTYPE* x;
  float* gamma;
  float* beta;
  PrepareAlloc(&x, N * C * D);
  PrepareAlloc(&gamma, C);
  PrepareAlloc(&beta, C);

  DTYPE* y;
  float* cache_ivar;
  float* cache_mean;
  PrepareAlloc(&y, N * C * D);
  PrepareAlloc(&cache_ivar, N * C);
  PrepareAlloc(&cache_mean, N * C);

  const float epsilon = 0.001f;
  InstanceNormGPU(x, gamma, beta, epsilon, N, C, D, y, cache_mean, cache_ivar);
  if (allow_print) {
    Print2D(y, N, C, D, "GPU y:");
  }

  DTYPE* y_h = new DTYPE[N * D];
  InstanceNormCPUHelper(x, gamma, beta, N, C, D, epsilon, y_h);
  if (allow_print) {
    Print2DHost(y_h, N, C, D, "CPU y:");
  }
  IsClose2D(y, y_h, N, C, D, "y");
  delete[] y_h;
  // ---- Forward Done Here ----

  // DTYPE* dy;
  // PrepareAlloc(&dy, N * D, 1);

  // DTYPE* dx;
  // float* dgamma;
  // float* dbeta;
  // PrepareAlloc(&dx, N * D);
  // PrepareAlloc(&dgamma, D);
  // PrepareAlloc(&dbeta, D);

  // LayerNormGradGPU(dy, x, cache_mean, cache_ivar, gamma, N, D, dx, dgamma,
  //                  dbeta);
  // if (allow_print) {
  //   Print2D(dgamma, 1, D, "GPU dgamma:");
  //   Print2D(dbeta, 1, D, "GPU dbeta:");
  //   Print2D(dx, N, D, "GPU dx:");
  // }

  // DTYPE* dx_h = new DTYPE[N * D];
  // float* dgamma_h = new float[D];
  // float* dbeta_h = new float[D];
  // LayerNormGradCPUHelper(dy, x, gamma, N, D, epsilon, dgamma_h, dbeta_h, dx_h);
  // if (allow_print) {
  //   Print2DHost(dgamma_h, 1, D, "CPU dgamma:");
  //   Print2DHost(dbeta_h, 1, D, "CPU dbeta:");
  //   Print2DHost(dx_h, N, D, "CPU dx:");
  // }

  // IsClose2D(dgamma, dgamma_h, 1, D, "dgamma");
  // IsClose2D(dbeta, dbeta_h, 1, D, "dbeta");
  // IsClose2D(dx, dx_h, N, D, "dx");

  // delete[] dx_h;
  // delete[] dgamma_h;
  // delete[] dbeta_h;
  // // ---- Backward Done Here ----

  checkCUDA(cudaFree(x));
  checkCUDA(cudaFree(gamma));
  checkCUDA(cudaFree(beta));
  checkCUDA(cudaFree(y));
  // checkCUDA(cudaFree(dy));
  // checkCUDA(cudaFree(dx));
  // checkCUDA(cudaFree(dgamma));
  // checkCUDA(cudaFree(dbeta));
  checkCUDA(cudaFree(cache_mean));
  checkCUDA(cudaFree(cache_ivar));
}
