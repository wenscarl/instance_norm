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

  double time_spent = 0.0;
  clock_t begin = clock();
  InstanceNormCPU(x_h, gamma_h, beta_h, N, C, D, epsilon, y_h);
  clock_t end = clock();
  time_spent += (double)(end - begin) / (CLOCKS_PER_SEC / 1000);
  printf("InstanceNormCPU time: %f ms\n", time_spent);

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

  printf("----------------------------------------------------------\n");
  double time_spent = 0.0;
  clock_t begin = clock();
  InstanceNormGradCPU(dy_h, x_h, gamma_h, N, C, D, epsilon, dgamma_h, dbeta_h, dx_h);
  clock_t end = clock();
  time_spent += (double)(end - begin) / (CLOCKS_PER_SEC / 1000);
  printf("InstanceNormGradCPU time: %f ms\n", time_spent);

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
  T* buf = new T[N * C * D];
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
  int C;
  int D;
  __device__ U Compute(const T* dy, const int& row, const int& col) const {
    U curr = GetAs<T, U>(dy, row * D + col);
    return curr * gamma[row % C] * (x[row * D + col] - cache_mean[row]) * (-0.5) *
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
  int C;
  int D;
  __device__ U Compute(const T* dy, const int& row, const int& col,
                       const U& dl_dvar) const {
    U curr = GetAs<T, U>(dy, row * D + col);
    return -1. * curr * gamma[row % C] * cache_ivar[row] +
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
  int C;
  int D;
  __device__ T Compute(const T* dy, const int& row, const int& col) const {
    U curr = GetAs<T, U>(dy, row * D + col);
    U dl_di = curr * gamma[row % C] * cache_ivar[row];
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
  int C;
  int D;
  __device__ T Compute(const T* x, const int& row, const int& col) const {
    U mean = cache_mean[row];
    U ivar = cache_ivar[row];
    U curr = GetAs<T, U>(x, row * D + col);
    return static_cast<T>((curr - mean) * ivar * gamma[row  % C] + beta[row  % C]);
  }
};

template <typename T, typename Op>
__global__ void InstanceNormUpdate(const T* __restrict__ in, const int N, 
                                const int D, T* out, Op op) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= N) return;

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
  printf("Now use_single_block=%d\n", use_single_warp);

  MeanOp<T, U> mean_ops{D};
  IvarOp<T, U> ivar_ops{cache_mean, D, epsilon};

  cudaEventRecord(start);
  int NxC = N * C;
  if (use_single_warp) {
    printf("XLOG: Mean/Var -> single-warp per row\n");
    printf("Run CUDA with %d blocks and each %d threads!\n", DivUp(NxC, kBlockSize / kWarpSize), kBlockSize);
    InstanceNormRowReduceInToOutWarp<<<DivUp(NxC, kBlockSize / kWarpSize),
                                    kBlockSize>>>(
        x, N, C, D, cache_mean, cache_ivar, mean_ops, ivar_ops);
  } else if (use_single_block) {
     printf("XLOG: Mean/Var -> single-block per row\n");
     InstanceNormRowReduceInToOut<<<N, kBlockSize>>>(
         x, N,C, D, cache_mean, cache_ivar, mean_ops, ivar_ops);
  } else {
    printf("XLOG: Mean/Var -> multi-block per row\n");
    const int blocks_per_row = DivUp(D, kBlockSize * min_workload_per_thread);

    float* temp_sum;
    float* temp_ivar;
    PrepareAlloc(&temp_sum, N * C * blocks_per_row);
    PrepareAlloc(&temp_ivar, N * C * blocks_per_row);

    dim3 threads(kBlockSize, 1, 1);
    dim3 blocks(blocks_per_row, N, 1);
    printf("XLOG: num_blocks per row=%d\n", blocks.x);

    // For long rows, we launch n blocks to process each row. The intermediate
    // results are stored in a temp memory with the size of N*n. Then, we launch
    // single block to handle each row of the temp memory.
    InstanceNormRowReduceInToTemp<<<blocks, threads>>>(x, N, C, D, temp_sum,
                                                    mean_ops);
    InstanceNormRowReduceTempToOut<<<N*C, threads>>>(temp_sum, N, C, blocks_per_row,
                                                cache_mean, mean_ops);

    InstanceNormRowReduceInToTemp<<<blocks, threads>>>(x, N, C, D, temp_ivar,
                                                    ivar_ops);
    InstanceNormRowReduceTempToOut<<<N*C, threads>>>(temp_ivar, N,C, blocks_per_row,
                                                cache_ivar, ivar_ops);

    checkCUDA(cudaFree(temp_ivar));
    checkCUDA(cudaFree(temp_sum));
  }
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds_reduce = 0;
  cudaEventElapsedTime(&milliseconds_reduce, start, stop);

  cudaEventRecord(start);
  YOp<T, U> y_ops{cache_mean, cache_ivar, gamma, beta, C, D};
  InstanceNormUpdate<<<DivUp(N*C*D, kBlockSize), kBlockSize>>>(x, N*C*D, D, y, y_ops);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds_update = 0;
  cudaEventElapsedTime(&milliseconds_update, start, stop);
  printf("InstanceNormGPU time %.2f ms (reduce=%f, update=%f)\n",
         milliseconds_reduce + milliseconds_update, milliseconds_reduce,
         milliseconds_update);
}

template <typename T, typename U, typename Op1, typename Op2>
__global__ void InstanceNormRowReduceInToOut(const T* __restrict__ in, const int N,
                                          const int D, U* out1, U* out2,
                                          Op1 op1, Op2 op2) {
  const int tid = threadIdx.x;

  typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ union {
    typename BlockReduce::TempStorage reduce;
    U broadcast[1];
  } temp_storage;

  for (int k = blockIdx.x; k < N; k += gridDim.x) {
    U partial_sum = 0;
    for (int i = tid; i < D; i += kBlockSize) {
      partial_sum += op1.Compute(in, k, i);
    }

    U sum = BlockReduce(temp_storage.reduce).Sum(partial_sum);

    if (tid == 0) {
      temp_storage.broadcast[0] = op1.Finalize(sum);
      out1[k] = op1.Finalize(sum);
    }
    __syncthreads();
    sum = temp_storage.broadcast[0];

    partial_sum = 0;
    for (int i = tid; i < D; i += kBlockSize) {
      partial_sum += op2.Compute(in, k, i, sum);
    }

    sum = BlockReduce(temp_storage.reduce).Sum(partial_sum);

    if (tid == 0) {
      out2[k] = op2.Finalize(sum);
    }
  }
}

template <typename T, typename U, typename Op1, typename Op2>
__global__ void InstanceNormRowReduceInToOutWarp(const T* __restrict__ in,
                                              const int N, const int D, U* out1,
                                              U* out2, Op1 op1, Op2 op2) {
  const int tid = threadIdx.x % kWarpSize;

  const int num_warps = kBlockSize / kWarpSize;
  typedef cub::WarpReduce<U> WarpReduce;
  typename WarpReduce::TempStorage temp_storage[num_warps];

  const int local_warp_id = threadIdx.x / kWarpSize;
  const int warp_id = blockIdx.x * num_warps + local_warp_id;

  for (int k = warp_id; k < N; k += gridDim.x * num_warps) {
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

template <typename T, typename U>
__global__ void InstanceNormGradBetaGammaIntoTemp(
    const T* __restrict__ dy, const T* __restrict__ x,
    const U* __restrict__ cache_mean, const U* __restrict__ cache_ivar,
    const int N, const int C, const int D, const int rows, U* __restrict__ tgamma,
    U* __restrict__ tbeta) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= C) return;
  int j = tid;

  U sum_dgamma = 0;
  U sum_dbeta = 0;
  int NxD = N * D;
  for (int i = blockIdx.y * rows; i < min(blockIdx.y * rows + rows, NxD); i++) {
    int in = i / D;
    int id = i % D;
    U dy_curr = GetAs<T, U>(dy, in * C * D + tid * D + id);
    sum_dgamma += dy_curr * (x[in * C * D + tid * D + id] - cache_mean[in*C + tid]) * cache_ivar[in*C + tid];
    sum_dbeta += dy_curr;
  }
  tgamma[blockIdx.y * C + j] = sum_dgamma;
  tbeta[blockIdx.y * C + j] = sum_dbeta;
}

template <typename U>
__global__ void InstanceNormGradBetaGammaTempToOut(const U* __restrict__ tg,
                                                  const U* __restrict__ tb,
                                                  const int C,
                                                  const int N,
                                                  U* __restrict__ dgamma,
                                                  U* __restrict__ dbeta) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // in reduced_rows per 128
  if (tid >= C) return;

  U sum_dgamma = 0;
  U sum_dbeta = 0;
  for (int i = 0 ; i < N; i++) {
    U tg_curr = tg[i * C + tid];
    U tb_curr = tb[i * C + tid];
    sum_dgamma += tg_curr;
    sum_dbeta += tb_curr;
  }

  dgamma[tid] = sum_dgamma;
  dbeta[tid] = sum_dbeta;
}

template <typename T, typename U>
__global__ void InstanceNormGradBetaGamma(
    const T* __restrict__ dy, const T* __restrict__ x,
    const U* __restrict__ cache_mean, const U* __restrict__ cache_ivar,
    const int N, const int C,  const int D, U* __restrict__ dgamma, U* __restrict__ dbeta) {
  // Assume the total thread number == D.
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= C) return;

  U sum_dgamma = 0;
  U sum_dbeta = 0;
  int NxD = N * D;
  for (int i = 0; i < NxD; i++) {
    int in = i/D;
    int id = i%D;
    U dy_curr = GetAs<T, U>(dy, in*C*D+tid*D+id);
    sum_dgamma += dy_curr * (x[in*C*D+tid*D+id] - cache_mean[in*C + tid]) * cache_ivar[in*C + tid];
    sum_dbeta += dy_curr;
  }
  dgamma[tid] = sum_dgamma;
  dbeta[tid] = sum_dbeta;
}

template <typename T, typename U>
void InstanceNormGradGPU(const T* dy, const T* x, const U* cache_mean,
                      const U* cache_ivar, const U* gamma, const int N, const int C,
                      const int D, T* dx, U* dgamma, U* dbeta) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int min_rows_per_block = 10000;
  bool use_temp_space = (N*D > min_rows_per_block);

  cudaEventRecord(start);

  if (!use_temp_space) {
    printf("XLOG: Dweight -> one block per column\n");
    printf("Run CUDA with %d blocks and each %d threads!\n", DivUp(C, kBlockSize), kBlockSize);
    InstanceNormGradBetaGamma<<<DivUp(C, kBlockSize), kBlockSize>>>(
        dy, x, cache_mean, cache_ivar,N,C,D, dgamma, dbeta);
  } else {
    printf("XLOG: Dweight -> multi-block per column\n");
    const int reduced_rows = DivUp(N*D, min_rows_per_block);
    float* temp_dgamma;
    float* temp_dbeta;
    PrepareAlloc(&temp_dgamma, reduced_rows * C);
    PrepareAlloc(&temp_dbeta, reduced_rows * C);

    dim3 blocks(DivUp(C, kBlockSize), reduced_rows);
    printf("XLOG: num_blocks per column=%d\n", blocks.y);
    InstanceNormGradBetaGammaIntoTemp<<<blocks, kBlockSize>>>(
        dy, x, cache_mean, cache_ivar, N, C, D, min_rows_per_block,
        temp_dgamma, temp_dbeta);
    InstanceNormGradBetaGammaTempToOut<<<DivUp(C, kBlockSize), kBlockSize>>>(
        temp_dgamma, temp_dbeta, C, reduced_rows, dgamma, dbeta);
    checkCUDA(cudaFree(temp_dgamma));
    checkCUDA(cudaFree(temp_dbeta));
  }
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds_dweight = 0;
  cudaEventElapsedTime(&milliseconds_dweight, start, stop);

  U* temp_1;  // dl_dvars
  U* temp_2;  // dl_dmus
  int NxC = N * C;
  PrepareAlloc(&temp_1, NxC);
  PrepareAlloc(&temp_2, NxC);

  bool use_single_warp = (D <= kWarpSize);

  const int min_num_blocks = kWarpSize;
  const int min_workload_per_thread = 50;
  bool use_single_block =
      (D <= min_num_blocks * kBlockSize * min_workload_per_thread);

  DvarOp<U, T> dl_dvar_ops{gamma, x, cache_ivar, cache_mean,C, D};
  DmeanOp<U, T> dl_dmu_ops{gamma, x, cache_ivar, cache_mean, temp_1,C,D};

  cudaEventRecord(start);
  if (use_single_warp) {
    printf("XLOG: Dvar/Dmean -> single-warp per row\n");
    InstanceNormRowReduceInToOutWarp<<<DivUp(NxC, kBlockSize / kWarpSize),
                                    kBlockSize>>>(dy, NxC, D, temp_1, temp_2,
                                                  dl_dvar_ops, dl_dmu_ops);
  } else if (use_single_block) {
    printf("XLOG: Dvar/Dmean -> single-block per row\n");
    printf("Run CUDA with %d blocks and each %d threads!\n", NxC, kBlockSize);
    InstanceNormRowReduceInToOut<<<NxC, kBlockSize>>>(dy, NxC, D, temp_1, temp_2,
                                                      dl_dvar_ops, dl_dmu_ops);
  } else {
    printf("XLOG: Dvar/Dmean -> multi-block per row\n");
    const int blocks_per_row = DivUp(D, kBlockSize * min_workload_per_thread);

    float* temp_dl_dvars;
    float* temp_dl_dmus;
    PrepareAlloc(&temp_dl_dvars, NxC * blocks_per_row);
    PrepareAlloc(&temp_dl_dmus, NxC * blocks_per_row);

    dim3 threads(kBlockSize, 1, 1);
    dim3 blocks(blocks_per_row, N, 1);
    printf("XLOG: num_blocks per row=%d\n", blocks.x);
    // printf("Run CUDA with %d blocks and each %d threads!\n", NxC, kBlockSize);
    InstanceNormRowReduceInToTemp<<<blocks, threads>>>(dy, N, C, D, temp_dl_dvars,
                                                    dl_dvar_ops);
    InstanceNormRowReduceTempToOut<<<NxC, threads>>>(
        temp_dl_dvars, N, C, blocks_per_row, temp_1, dl_dvar_ops);

    InstanceNormRowReduceInToTemp<<<blocks, threads>>>(dy, N,C, D, temp_dl_dmus,
                                                    dl_dmu_ops);
    InstanceNormRowReduceTempToOut<<<NxC, threads>>>(temp_dl_dmus, N, C, blocks_per_row,
                                                temp_2, dl_dmu_ops);

    checkCUDA(cudaFree(temp_dl_dvars));
    checkCUDA(cudaFree(temp_dl_dmus));
    // printf("XLOG: Dvar/Dmean -> multi-block per row\n");
    // const int blocks_per_row = DivUp(D, kBlockSize * min_workload_per_thread);

    // float* temp_dl_dvars;
    // float* temp_dl_dmus;
    // PrepareAlloc(&temp_dl_dvars, N * blocks_per_row);
    // PrepareAlloc(&temp_dl_dmus, N * blocks_per_row);

    // dim3 threads(kBlockSize, 1, 1);
    // dim3 blocks(blocks_per_row, N, 1);
    // printf("XLOG: num_blocks per row=%d\n", blocks.x);

    // InstanceNormRowReduceInToTemp<<<blocks, threads>>>(dy, N, C, D, temp_dl_dvars,
    //                                                 dl_dvar_ops);
    // InstanceNormRowReduceTempToOut<<<N, threads>>>(
    //     temp_dl_dvars, N, C, blocks_per_row, temp_1, dl_dvar_ops);

    // InstanceNormRowReduceInToTemp<<<blocks, threads>>>(dy, N,C, D, temp_dl_dmus,
    //                                                 dl_dmu_ops);
    // InstanceNormRowReduceTempToOut<<<N, threads>>>(temp_dl_dmus, N, C, blocks_per_row,
    //                                             temp_2, dl_dmu_ops);

    // checkCUDA(cudaFree(temp_dl_dvars));
    // checkCUDA(cudaFree(temp_dl_dmus));
  }
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds_reduce = 0;
  cudaEventElapsedTime(&milliseconds_reduce, start, stop);

  cudaEventRecord(start);
  DxOp<T, U> dx_ops{x, cache_mean, cache_ivar, gamma, temp_1, temp_2, C, D};
  InstanceNormUpdate<<<DivUp(NxC*D, kBlockSize), kBlockSize>>>(dy, NxC*D,  D, dx,
                                                            dx_ops);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds_update = 0;
  cudaEventElapsedTime(&milliseconds_update, start, stop);
  printf("InstanceNormGradGPU time %.2f ms (dweight=%f, reduce=%f, update=%f)\n",
         milliseconds_dweight + milliseconds_reduce + milliseconds_update,
         milliseconds_dweight, milliseconds_reduce, milliseconds_update);

  checkCUDA(cudaFree(temp_1));
  checkCUDA(cudaFree(temp_2));
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

template <typename T, typename U, typename Op1, typename Op2>
__global__ void InstanceNormRowReduceInToOut(const T* __restrict__ in, const int N, const int C,
                                             const int D, U* out1, U* out2,
                                             Op1 op1, Op2 op2) {
  const int tid = threadIdx.x;

  typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ union {
    typename BlockReduce::TempStorage reduce;
    U broadcast[1];
  } temp_storage;

  for (int k = blockIdx.x; k < N*C; k += gridDim.x) {
    U partial_sum = 0;
    for (int i = tid; i < D; i += kBlockSize) {
      partial_sum += op1.Compute(in, k, i);
    }

    U sum = BlockReduce(temp_storage.reduce).Sum(partial_sum);

    if (tid == 0) {
      temp_storage.broadcast[0] = op1.Finalize(sum);
      out1[k] = op1.Finalize(sum);
    }
    __syncthreads();
    sum = temp_storage.broadcast[0];

    partial_sum = 0;
    for (int i = tid; i < D; i += kBlockSize) {
      partial_sum += op2.Compute(in, k, i, sum);
    }

    sum = BlockReduce(temp_storage.reduce).Sum(partial_sum);

    if (tid == 0) {
      out2[k] = op2.Finalize(sum);
    }
  }
}

template <typename T, typename U, typename Op>
__global__ void InstanceNormRowReduceInToTemp(const T* __restrict__ x, const int N, const int C,
                                           const int D, U* __restrict__ temp,
                                           Op op) {
  typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;

  for (int row_idx = blockIdx.y; row_idx < N*C; row_idx += gridDim.y) {
    U partial_sum = 0;
    for (int i = row_offset; i < D; i += gridDim.x * blockDim.x) {
      partial_sum += op.Compute(x, row_idx, i);
    }
    U sum = BlockReduce(temp_storage).Sum(partial_sum);
    if (threadIdx.x == 0) {
      temp[row_idx * gridDim.x + blockIdx.x] = sum;
    }
  }
}

template <typename U, typename Op>
__global__ void InstanceNormRowReduceTempToOut(const U* __restrict__ temp,
                                            const int N, const int C, const int cols,
                                            U* __restrict__ cache, Op op) {
  typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int k = blockIdx.x; k < N*C; k += gridDim.x) {
    U partial_sum = 0;
    for (int i = threadIdx.x; i < cols; i += kBlockSize) {
      partial_sum += temp[k * cols + i];
    }

    U sum = BlockReduce(temp_storage).Sum(partial_sum);

    if (threadIdx.x == 0) {
      cache[k] = op.Finalize(sum);
    }
  }
}

#define DTYPE float

int main(int argc, char** argv) {
  /** Parameters and Knobs **/
  int N = 2;
  int C = 3;
  int D = 4;
  int allow_print = 0;
  if (argc >= 3) {
    N = atoi(argv[1]);
    C = atoi(argv[2]);
    D = atoi(argv[3]);
    allow_print = atoi(argv[4]);
  }

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
 //   Print2D(x, N, C, D, "GPU x:");
 //   Print2D(y, N, C, D, "GPU y:");
    //Print2D(cache_mean, N, C, 1, "GPU cache_mean:");
    //Print2D(cache_ivar, N, C, 1, "GPU cache_ivar:");
  }

  DTYPE* y_h = new DTYPE[N * C * D];
  InstanceNormCPUHelper(x, gamma, beta, N, C, D, epsilon, y_h);
  if (allow_print) {
 //  Print2DHost(y_h, N, C, D, "CPU y:");
  }
  IsClose2D(y, y_h, N, C, D, "y");
  delete[] y_h;
  // ---- Forward Done Here ----

  DTYPE* dy;
  PrepareAlloc(&dy, N * C * D, 0.0001f);

  DTYPE* dx;
  float* dgamma;
  float* dbeta;
  PrepareAlloc(&dx, N * C* D);
  PrepareAlloc(&dgamma, C);
  PrepareAlloc(&dbeta, C);

  printf("---------------------------------Starting of InstanceNormGradGPU!-------------------------------------\n");
  InstanceNormGradGPU(dy, x, cache_mean, cache_ivar, gamma, N, C, D, dx, dgamma,
                      dbeta);
  if (allow_print) {
    Print2D(dgamma, 1, 1, C, "GPU dgamma:");
    Print2D(dbeta, 1, 1, C, "GPU dbeta:");
    Print2D(dx, N, C, D, "GPU dx:");
 //    Print2D(dy, N, C, D, "GPU dy:");
  }

  DTYPE* dx_h = new DTYPE[N * C * D];
  float* dgamma_h = new float[C];
  float* dbeta_h = new float[C];
  InstanceNormGradCPUHelper(dy, x, gamma, N, C, D, epsilon, dgamma_h, dbeta_h, dx_h);
  if (allow_print) {
    Print2DHost(dgamma_h, 1, 1, C, "CPU dgamma:");
    Print2DHost(dbeta_h, 1, 1, C, "CPU dbeta:");
    Print2DHost(dx_h, N, C, D, "CPU dx:");
  }

  IsClose2D(dgamma, dgamma_h, 1, 1, C, "dgamma");
  IsClose2D(dbeta, dbeta_h, 1, 1, C, "dbeta");
  IsClose2D(dx, dx_h, N, C, D, "dx");

  delete[] dx_h;
  delete[] dgamma_h;
  delete[] dbeta_h;
  // // ---- Backward Done Here ----

  checkCUDA(cudaFree(x));
  checkCUDA(cudaFree(gamma));
  checkCUDA(cudaFree(beta));
  checkCUDA(cudaFree(y));
  checkCUDA(cudaFree(dy));
  checkCUDA(cudaFree(dx));
  checkCUDA(cudaFree(dgamma));
  checkCUDA(cudaFree(dbeta));
  checkCUDA(cudaFree(cache_mean));
  checkCUDA(cudaFree(cache_ivar));
}
