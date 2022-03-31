#include <cub/block/block_reduce.cuh>
//#include "fp16_emu.h"
#include <iostream>

#define checkCUDA(expression)                               \
  {                                                         \
    cudaError_t status = (expression);                      \
    if (status != cudaSuccess)                              \
    {                                                       \
      std::cerr << "Error on line " << __LINE__ << ": "     \
                << cudaGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                              \
    }                                                       \
  }

#define min(a, b)           \
  ({                        \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a < _b ? _a : _b;      \
  })

template <typename T>
void IsClose2DHost(const T *x, const T *y, int N, int C, int D, std::string msg,
                   float atol, float rtol);

template <typename T>
void Print2DHost(const T *x, int N, int C, int D, std::string msg);

template <typename T, typename U>
void InstanceNormCPU(const T *x, const U *gamma, const U *beta, const int N,
                     const int C, const int D, const U epsilon, T *y,
                     U *cache_mean_cpu, U *cache_ivar_cpu,
                     const int is_channel_first);

template <typename T, typename U>
void InstanceNormGradCPU(const T *dy, const T *x, const U *gamma, const int N,
                         const int C, const int D, const U epsilon, U *dgamma,
                         U *dbeta, T *dx, U *dl_dvars, U *dl_dmus,
                         const int is_channel_first);

template <typename T, typename U>
void InstanceNormCPUHelper(const T *x, const U *gamma, const U *beta,
                           const int N, const int C, const int D,
                           const U epsilon, T *y_h, U *cache_mean_cpu,
                           U *cache_ivar_cpu, const int is_channel_first)
{
  T *x_h = new T[N * C * D];
  U *gamma_h = new U[C];
  U *beta_h = new U[C];

  checkCUDA(cudaMemcpy(x_h, x, N * C * D * sizeof(T), cudaMemcpyDeviceToHost));
  checkCUDA(cudaMemcpy(gamma_h, gamma, C * sizeof(U), cudaMemcpyDeviceToHost));
  checkCUDA(cudaMemcpy(beta_h, beta, C * sizeof(U), cudaMemcpyDeviceToHost));

  double time_spent = 0.0;
  clock_t begin = clock();
  InstanceNormCPU(x_h, gamma_h, beta_h, N, C, D, epsilon, y_h, cache_mean_cpu,
                  cache_ivar_cpu, is_channel_first);
  clock_t end = clock();
  time_spent += (double)(end - begin) / (CLOCKS_PER_SEC / 1000);
  printf("InstanceNormCPU time: %f ms\n", time_spent);

  delete[] x_h;
  delete[] gamma_h;
  delete[] beta_h;
}

template <typename T, typename U>
void InstanceNormGradCPUHelper(const T *dy, const T *x, const U *gamma,
                               const int N, const int C, const int D,
                               const U epsilon, U *dgamma_h, U *dbeta_h,
                               T *dx_h, U *dl_dvars_h, U *dl_dmus_h,
                               const int is_channel_first)
{
  T *dy_h = new T[N * C * D];
  T *x_h = new T[N * C * D];
  U *gamma_h = new U[C];
  checkCUDA(
      cudaMemcpy(dy_h, dy, N * C * D * sizeof(T), cudaMemcpyDeviceToHost));
  checkCUDA(cudaMemcpy(x_h, x, N * C * D * sizeof(T), cudaMemcpyDeviceToHost));
  checkCUDA(cudaMemcpy(gamma_h, gamma, C * sizeof(U), cudaMemcpyDeviceToHost));

  printf("----------------------------------------------------------\n");
  double time_spent = 0.0;
  clock_t begin = clock();
  InstanceNormGradCPU(dy_h, x_h, gamma_h, N, C, D, epsilon, dgamma_h, dbeta_h,
                      dx_h, dl_dvars_h, dl_dmus_h, is_channel_first);
  clock_t end = clock();
  time_spent += (double)(end - begin) / (CLOCKS_PER_SEC / 1000);
  printf("InstanceNormGradCPU time: %f ms\n", time_spent);

  delete[] dy_h;
  delete[] x_h;
  delete[] gamma_h;
}

const int kBlockSize = 256;
const int kWarpSize = 32;

int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void overwrite(T *cache_mean, T *cache_ivar, T *cache_mean_cpu,
               T *cache_ivar_cpu, int size)
{
  checkCUDA(cudaMemcpy(cache_mean, cache_mean_cpu, size * sizeof(T),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(cache_ivar, cache_ivar_cpu, size * sizeof(T),
                       cudaMemcpyHostToDevice));
}

template <typename T>
void PrepareAlloc(T **x, int size, int init = -1)
{
  srand(12);
  T *buf = new T[size];
  for (int i = 0; i < size; i++)
  {
    if (init != -1)
    {
      buf[i] = init;
    }
    else
    {
      T HI = 1;
      T LO = 0;
      buf[i] = LO + static_cast<T>(static_cast<float>(rand()) /
                                   (RAND_MAX / (HI - LO)));
    }
  }

  checkCUDA(cudaMalloc(&(*x), size * sizeof(T)));
  checkCUDA(cudaMemcpy(*x, buf, size * sizeof(T), cudaMemcpyHostToDevice));

  delete[] buf;
}

template <typename T>
void Print2D(const T *x, int N, int C, int D, std::string msg)
{
  T *buf = new T[N * C * D];
  checkCUDA(cudaMemcpy(buf, x, N * C * D * sizeof(T), cudaMemcpyDeviceToHost));
  Print2DHost(buf, N, C, D, msg);
  delete[] buf;
}

template <typename T>
void IsClose2D(const T *x, const T *y, int N, int C, int D, std::string msg,
               float atol = 1e-3, float rtol = 1e-3)
{
  if (D == 10000000)
  { // Mainly for y when NxD=10x10000000
    atol = 1e-1;
  }
  if (D == 1000000)
  { // Mainly for y when NxD=100x1000000
    atol = 1e-2;
  }
  if (D == 10)
  { // Mainly for dgamma when NxD=10000000x10
    atol = 1e-1;
  }
  T *buf = new T[N * C * D];
  checkCUDA(cudaMemcpy(buf, x, N * C * D * sizeof(T), cudaMemcpyDeviceToHost));
  IsClose2DHost(buf, y, N, C, D, msg, atol, rtol);
  delete[] buf;
}

template <typename T, typename U>
__host__ __device__ U GetAs(const T *__restrict__ in, int offset)
{
  return static_cast<U>(in[offset]);
}
///////////////////////////// MAYBE inspect
template <typename T, typename U>
struct MeanOp
{
  int D;
  int C = -1; // C==-1 indicating NCD
  __device__ U Compute(const T *x, const int &row, const int &col) const
  {
    int idx = (C == -1) ? row * D + col : (row / C) * C * D + col * C + row % C;
    return GetAs<T, U>(x, idx);
  }
  __device__ U Compute_ndc(const T *x, const int &row, const int &col,
                           const int &z) const
  {
    int idx = row * C * D + col * C + z;
    return GetAs<T, U>(x, idx);
  }
  __device__ U Finalize(const U &sum) const { return sum / D; }
};

template <typename T, typename U>
struct IvarOp
{
  const U *cache_mean;
  U epsilon;
  int D;
  int C = -1; // C==-1 indicating NCD
  __device__ U Compute(const T *x, const int &row, const int &col,
                       const U &mean) const
  {
    int idx = (C == -1) ? row * D + col : (row / C) * C * D + col * C + row % C;
    U curr = GetAs<T, U>(x, idx);
    return (curr - mean) * (curr - mean);
  }
  __device__ U Compute(const T *x, const int &row, const int &col) const
  {
    return Compute(x, row, col, cache_mean[row]);
  }
  __device__ U Compute_ndc(const T *x, const int &row, const int &col,
                           const int &z) const
  {
    int idx = row * C * D + col * C + z;
    U curr = GetAs<T, U>(x, idx);
    U m = cache_mean[row * C + z];
    return (curr - m) * (curr - m);
  }
  __device__ U Compute_ndc(const T *x, const int &row, const int &col,
                           const int &z, const U &mean) const
  {
    int idx = row * C * D + col * C + z;
    U curr = GetAs<T, U>(x, idx);
    return (curr - mean) * (curr - mean);
  }
  __device__ U Finalize(const U &sum) const { return rsqrt(sum / D + epsilon); }
};

template <typename T, typename U>
struct DvarOp
{
  const U *gamma;
  const T *x;
  const U *cache_ivar;
  const U *cache_mean;
  int C;
  int D;
  __device__ U Compute(const T *dy, const int &row, const int &col) const
  {
    U curr = GetAs<T, U>(dy, row * D + col);
    return curr * gamma[row % C] * (x[row * D + col] - cache_mean[row]) *
           (-0.5) * (cache_ivar[row] * cache_ivar[row] * cache_ivar[row]);
  }
  __device__ U Compute_ndc(const T *dy, const int &row, const int &col,
                           const int &z) const
  {
    U curr = GetAs<T, U>(dy, row * D * C + col * C + z);
    U ivar = cache_ivar[row * C + z];
    U me = cache_mean[row * C + z];
    return curr * gamma[z] * (x[row * D * C + col * C + z] - me) * (-0.5) *
           (ivar * ivar * ivar);
  }
  __device__ U Finalize(const U &sum) const { return sum; }
};

template <typename T, typename U>
struct DmeanOp
{
  const U *gamma;
  const T *x;
  const U *cache_ivar;
  const U *cache_mean;
  const U *dl_dvars;
  int C;
  int D;
  __device__ U Compute(const T *dy, const int &row, const int &col,
                       const U &dl_dvar) const
  {
    U curr = GetAs<T, U>(dy, row * D + col);
    return -1. * curr * gamma[row % C] * cache_ivar[row] +
           dl_dvar * (-2. / D) * (x[row * D + col] - cache_mean[row]);
  }
  __device__ U Compute(const T *dy, const int &row, const int &col) const
  {
    return Compute(dy, row, col, dl_dvars[row]);
  }

  __device__ U Compute_ndc(const T *dy, const int &row, const int &col,
                           const int &z) const
  {
    U curr = GetAs<T, U>(dy, row * D * C + col * C + z);
    U ivar = cache_ivar[row * C + z];
    U me = cache_mean[row * C + z];
    return -1. * curr * gamma[z] * ivar +
           dl_dvars[row * C + z] * (-2. / D) *
               (x[row * D * C + col * C + z] - me);
  }

  __device__ U Compute_ndc(const T *dy, const int &row, const int &col,
                           const int &z, const U &dl_dvars) const
  {
    U curr = GetAs<T, U>(dy, row * D * C + col * C + z);
    U ivar = cache_ivar[row * C + z];
    U me = cache_mean[row * C + z];
    return -1. * curr * gamma[z] * ivar +
           dl_dvars * (-2. / D) * (x[row * D * C + col * C + z] - me);
  }

  __device__ U Finalize(const U &sum) const { return sum; }
};

template <typename T, typename U>
struct DxOp
{
  const T *x;
  const U *cache_mean;
  const U *cache_ivar;
  const U *gamma;
  const U *dl_dvars;
  const U *dl_dmus;
  int C;
  int D;
  __device__ T Compute(const T *dy, const int &idx,
                       const int is_channel_first) const
  {
    U curr = GetAs<T, U>(dy, idx);
    U dl_dx;
    if (is_channel_first)
    { // NCD
      int row = idx / D;
      U dl_di = curr * gamma[row % C] * cache_ivar[row];
      U di_dx = 1.;
      U dvar_dx = 2. * (x[idx] - cache_mean[row]) / D;
      U dmu_dx = 1. / D;
      dl_dx = dl_di * di_dx + dl_dvars[row] * dvar_dx + dl_dmus[row] * dmu_dx;
    }
    else
    { // NDC
      int col = idx % C;
      int cache_idx = idx / (C * D) * C + idx % C;
      U dl_di = curr * gamma[col] * cache_ivar[cache_idx];
      U di_dx = 1.;
      U dvar_dx = 2. * (x[idx] - cache_mean[cache_idx]) / D;
      U dmu_dx = 1. / D;
      dl_dx = dl_di * di_dx  + dl_dvars[cache_idx] * dvar_dx + dl_dmus[cache_idx] * dmu_dx;
      // dl_dx = dl_di * di_dx ;
    }
    return static_cast<T>(dl_dx);
  }
};

template <typename T, typename U>
struct YOp
{
  const U *cache_mean;
  const U *cache_ivar;
  const U *gamma;
  const U *beta;
  int C;
  int D;
  __device__ T Compute(const T *x, const int &idx,
                       const int is_channel_first) const
  {
    U curr = GetAs<T, U>(x, idx);
    int gb_idx, cache_idx;
    if (is_channel_first)
    {
      gb_idx = (idx / D) % C;
      cache_idx = idx / D;
    }
    else
    {
      gb_idx = idx % C;
      cache_idx = (idx / (C * D)) * C + gb_idx;
    }
    U mean = cache_mean[cache_idx];
    U ivar = cache_ivar[cache_idx];
    return static_cast<T>((curr - mean) * ivar * gamma[gb_idx] + beta[gb_idx]);
    // return static_cast<T>((curr) * gamma[gb_idx] + beta[gb_idx]);
  }
};

template <typename T, typename U, typename Op>
__global__ void InstanceNormRowReduceInToOut_NDC(const T *__restrict__ in,
                                                 const int N, const int C,
                                                 const int D,
                                                 U *__restrict__ temp, Op op)
{
  // blocks(C / 32, 1, N);
  // threads(32, kBlocksize/32,1)
  const int x_tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int y_tid = threadIdx.y + blockIdx.y * blockDim.y;
  const int z_tid = threadIdx.z + blockIdx.z * blockDim.z;
  if (x_tid >= C)
    return;
  U partial_sum = 0;
  for (int i = y_tid; i < D; i += blockDim.y * gridDim.y)
  {
    partial_sum += op.Compute_ndc(in, z_tid, i, x_tid);
  }

  temp[(z_tid * C + x_tid) * blockDim.y * gridDim.y + y_tid] = partial_sum;
}

template <typename T, typename Op>
__global__ void InstanceNormUpdate(const T *__restrict__ in, const int N,
                                   const int D, T *out, Op op,
                                   const int is_channel_first = 1)
{
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= N)
    return;
  for (int row_idx = tid; row_idx < N; row_idx += gridDim.x * blockDim.x)
  {
    out[row_idx] = op.Compute(in, row_idx, is_channel_first);
  }
}

template <typename T, typename U>
void InstanceNormGPU(const T *x, const U *gamma, const U *beta, const U epsilon,
                     const int N, const int C, const int D, T *y, U *cache_mean,
                     U *cache_ivar, const int is_channel_first)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  bool use_single_warp = (D <= kWarpSize);

  const int min_num_blocks = kWarpSize;
  const int min_workload_per_thread = 100;
  bool use_single_block =
      (D <= min_num_blocks * kBlockSize * min_workload_per_thread);
  printf("InstanceNormGPU use_single_block=%d\n", use_single_block);

  int maybe_channel_dim = is_channel_first ? -1 : C;
  MeanOp<T, U> mean_ops{D, maybe_channel_dim};
  IvarOp<T, U> ivar_ops{cache_mean, epsilon, D, maybe_channel_dim};

  cudaEventRecord(start);
  int NxC = N * C;
  if (use_single_warp)
  {
    printf("XLOG: Mean/Var -> single-warp per row\n");
    printf("Run CUDA with %d blocks and each %d threads!\n",
           DivUp(NxC, kBlockSize / kWarpSize), kBlockSize);
    InstanceNormRowReduceInToOutWarp<<<DivUp(NxC, kBlockSize / kWarpSize),
                                       kBlockSize>>>(
        x, N, C, D, cache_mean, cache_ivar, mean_ops, ivar_ops,
        is_channel_first);
  }
  else if (use_single_block)
  {
    printf("XLOG: Mean/Var -> single-block per row\n");
    if (is_channel_first)
    {
      InstanceNormRowReduceInToOut<<<N, kBlockSize>>>(
          x, N, C, D, cache_mean, cache_ivar, mean_ops, ivar_ops,
          is_channel_first);
    }
    else
    {

      // blocks(C / 32, N, 1);
      // threads(32,1, 128/32)
      float *temp_sum, *temp_ivar;
      dim3 threads(kWarpSize, kBlockSize / kWarpSize);
      const int local_min_workload_per_thread = 3200;
      int ppr = DivUp(D, threads.y * local_min_workload_per_thread); //~ 10

      dim3 blocks(DivUp(C, kWarpSize), ppr, N);

      printf("XLOG: in NDC, num_blocks per row=%d\n", ppr);
      printf("XLOG: thread.XYZ=%d, %d, %d\n", threads.x, threads.y, threads.z);
      printf("XLOG: blocks.XYZ=%d, %d, %d\n", blocks.x, blocks.y, blocks.z);
      PrepareAlloc(&temp_sum, N * C * threads.y * ppr);
      PrepareAlloc(&temp_ivar, N * C * threads.y * ppr);
      InstanceNormRowReduceInToOut_NDC<<<blocks, threads>>>(x, N, C, D,
                                                            temp_sum, mean_ops);
      InstanceNormRowReduceTempToOut<<<N * C, kBlockSize>>>(
          temp_sum, N, C, threads.y * ppr, cache_mean, mean_ops);

      InstanceNormRowReduceInToOut_NDC<<<blocks, threads>>>(
          x, N, C, D, temp_ivar, ivar_ops);
      InstanceNormRowReduceTempToOut<<<N * C, kBlockSize>>>(
          temp_ivar, N, C, threads.y * ppr, cache_ivar, ivar_ops);

      checkCUDA(cudaFree(temp_sum));
      checkCUDA(cudaFree(temp_ivar));
    }
  }
  else
  {
    printf("XLOG: Mean/Var -> multi-block per row\n");
    const int blocks_per_row = DivUp(D, kBlockSize * min_workload_per_thread);

    float *temp_sum;
    float *temp_ivar;
    dim3 threads, blocks;
    int temp_total_rows;
    printf("XLOG: I am %d !\b", is_channel_first);
    if (is_channel_first)
    {
      PrepareAlloc(&temp_sum, N * C * blocks_per_row);
      PrepareAlloc(&temp_ivar, N * C * blocks_per_row);
      threads.x = kBlockSize;
      blocks.x = blocks_per_row;
      blocks.y = N;
      temp_total_rows = blocks_per_row;
      printf("XLOG: num_blocks per row=%d\n", blocks.x);
      // For long rows, we launch n blocks to process each row. The intermediate
      // results are stored in a temp memory with the size of N*n. Then, we
      // launch single block to handle each row of the temp memory.
    }
    else
    {
      int thd_x = min(N * C, kBlockSize);
      int thd_y = kBlockSize / thd_x;
      int min_workload_per_thread = 10000;
      const int blocks_per_row = DivUp(D, thd_y * min_workload_per_thread);

      threads.x = thd_x;
      threads.y = thd_y;
      blocks.x = DivUp(N * C, thd_x);
      blocks.y = blocks_per_row;
      temp_total_rows = blocks_per_row * thd_y;
      PrepareAlloc(&temp_sum, N * C * blocks_per_row * thd_y);
      PrepareAlloc(&temp_ivar, N * C * blocks_per_row * thd_y);

      printf("XLOG: Mean/Var -> multi-block per row\n");
      printf("XLOG: in NDC, num_blocks per row=%d\n", blocks_per_row);
      printf("XLOG: thread.XYZ=%d, %d, %d\n", threads.x, threads.y, threads.z);
      printf("XLOG: blocks.XYZ=%d, %d, %d\n", blocks.x, blocks.y, blocks.z);

      // For long rows, we launch n blocks to process each row. The intermediate
      // results are stored in a temp memory with the size of N*n. Then, we
      // launch single block to handle each row of the temp memory.
    }
    InstanceNormRowReduceInToTemp<<<blocks, threads>>>(
        x, N, C, D, temp_sum, mean_ops, is_channel_first);
    InstanceNormRowReduceTempToOut<<<N * C, kBlockSize>>>(
        temp_sum, N, C, temp_total_rows, cache_mean, mean_ops);

    InstanceNormRowReduceInToTemp<<<blocks, threads>>>(
        x, N, C, D, temp_ivar, ivar_ops, is_channel_first);
    InstanceNormRowReduceTempToOut<<<N * C, kBlockSize>>>(
        temp_ivar, N, C, temp_total_rows, cache_ivar, ivar_ops);

    checkCUDA(cudaFree(temp_ivar));
    checkCUDA(cudaFree(temp_sum));
  }
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds_reduce = 0;
  cudaEventElapsedTime(&milliseconds_reduce, start, stop);

  cudaEventRecord(start);
  YOp<T, U> y_ops{cache_mean, cache_ivar, gamma, beta, C, D};
  int min_work_per_thread = 100;
  InstanceNormUpdate<<<DivUp(N * C * D, kBlockSize * min_work_per_thread),
                       kBlockSize>>>(x, N * C * D, D, y, y_ops,
                                     is_channel_first);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds_update = 0;
  cudaEventElapsedTime(&milliseconds_update, start, stop);
  printf("InstanceNormGPU time %.2f ms (reduce=%f, update=%f)\n",
         milliseconds_reduce + milliseconds_update, milliseconds_reduce,
         milliseconds_update);
}

template <typename T, typename U, typename Op1, typename Op2>
__global__ void InstanceNormRowReduceInToOut(const T *__restrict__ in,
                                             const int N, const int D, U *out1,
                                             U *out2, Op1 op1, Op2 op2)
{
  const int tid = threadIdx.x;

  typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ union
  {
    typename BlockReduce::TempStorage reduce;
    U broadcast[1];
  } temp_storage;

  for (int k = blockIdx.x; k < N; k += gridDim.x)
  {
    U partial_sum = 0;
    for (int i = tid; i < D; i += kBlockSize)
    {
      partial_sum += op1.Compute(in, k, i);
    }

    U sum = BlockReduce(temp_storage.reduce).Sum(partial_sum);

    if (tid == 0)
    {
      temp_storage.broadcast[0] = op1.Finalize(sum);
      out1[k] = op1.Finalize(sum);
    }
    __syncthreads();
    sum = temp_storage.broadcast[0];

    partial_sum = 0;
    for (int i = tid; i < D; i += kBlockSize)
    {
      partial_sum += op2.Compute(in, k, i, sum);
    }

    sum = BlockReduce(temp_storage.reduce).Sum(partial_sum);

    if (tid == 0)
    {
      out2[k] = op2.Finalize(sum);
    }
  }
}

template <typename T, typename U>
__global__ void InstanceNormBetaGammaRowReduceInToTemp(
    const T *__restrict__ x, const T *__restrict__ dy,
    const U *__restrict__ cache_mean, const U *__restrict__ cache_ivar,
    const int N, const int C, const int D, U *__restrict__ temp_dbeta,
    U *__restrict__ temp_dgamma, const int is_channel_first = true)
{

  if (is_channel_first)
  {
    typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;
    const int col_offset = threadIdx.y + blockIdx.y * blockDim.y;

    int NxD = N * D;
    int CxD = C * D;
    int glb_id, cache_idx;
    for (int col_idx = col_offset; col_idx < C;
         col_idx += gridDim.y * blockDim.y)
    {
      U partial_sum_dbeta = 0;
      U partial_sum_dgamma = 0;
      for (int i = row_offset; i < NxD; i += gridDim.x * blockDim.x)
      {
        glb_id = (i / D) * CxD + col_idx * D + i % D;
        cache_idx = i / D * C + col_idx;
        U curr = dy[glb_id];
        partial_sum_dbeta += curr;
        partial_sum_dgamma +=
            curr * (x[glb_id] - cache_mean[cache_idx]) * cache_ivar[cache_idx];
      }
      U sum_dbeta = BlockReduce(temp_storage).Sum(partial_sum_dbeta);
      U sum_dgamma = BlockReduce(temp_storage).Sum(partial_sum_dgamma);
      if (threadIdx.x == 0)
      {
        temp_dbeta[blockIdx.x * C + col_idx] = sum_dbeta;
        temp_dgamma[blockIdx.x * C + col_idx] = sum_dgamma;
      }
    }
  }
  else
  {
    // blocks: 1 , n_rows, 1
    // 64 thds
    typedef cub::BlockReduce<U, 64> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;
    const int col_offset = threadIdx.y + blockIdx.y * blockDim.y;
    if (row_offset >= C)
      return;
    if (col_offset >= N * D)
      return;
    int NxD = N * D;
    int CxD = C * D;
    int glb_id, cache_idx;

    // printf("here %d and %d\n",gridDim.y * blockDim.y,col_offset);
    for (int row_idx = row_offset; row_idx < C;
         row_idx += blockDim.x * gridDim.x)
    {
      U partial_sum_dbeta = 0;
      U partial_sum_dgamma = 0;
      for (int i = col_offset; i < NxD; i += gridDim.y * blockDim.y)
      {
        int row_idx = row_offset;
        glb_id = i * C + row_idx;
        cache_idx = i / D * C + row_idx;
        U curr = dy[glb_id];
        partial_sum_dbeta += curr; // cache_mean[cache_idx];
        partial_sum_dgamma +=
            curr * (x[glb_id] - cache_mean[cache_idx]) * cache_ivar[cache_idx];
      }
      temp_dbeta[col_offset * C + row_idx] = partial_sum_dbeta;
      temp_dgamma[col_offset * C + row_idx] = partial_sum_dgamma;
    }
    // printf("%d\n",blockIdx.y);

    // temp_dbeta[ blockIdx.y * C + row_offset] = partial_sum_dbeta;
    // temp_dgamma[ blockIdx.y * C + row_offset] = partial_sum_dgamma;
  }
}

template <typename T, typename U>
__global__ void InstanceNormGradBetaGammaIntoTemp(
    const T *__restrict__ dy, const T *__restrict__ x,
    const U *__restrict__ cache_mean, const U *__restrict__ cache_ivar,
    const int N, const int C, const int D, const int rows,
    U *__restrict__ tgamma, U *__restrict__ tbeta)
{
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= C)
    return;
  int j = tid;

  U sum_dgamma = 0;
  U sum_dbeta = 0;
  int NxD = N * D;
  for (int i = blockIdx.y * rows; i < min(blockIdx.y * rows + rows, NxD); i++)
  {
    int in = i / D;
    int id = i % D;
    // NCD
    U dy_curr = GetAs<T, U>(dy, in * C * D + tid * D + id);
    sum_dgamma += dy_curr *
                  (x[in * C * D + tid * D + id] - cache_mean[in * C + tid]) *
                  cache_ivar[in * C + tid];
    // NDC
    //       U dy_curr = GetAs<T, U>(dy, tid  + i * C);
    //       sum_dgamma += dy_curr * (x[tid  + i*C] - cache_mean[in + tid*N]) *
    //       cache_ivar[in + tid*N];

    sum_dbeta += dy_curr;
  }
  tgamma[blockIdx.y * C + j] = sum_dgamma;
  tbeta[blockIdx.y * C + j] = sum_dbeta;
}

template <typename U>
__global__ void InstanceNormGradBetaGammaTempToOut(const U *__restrict__ tg,
                                                   const U *__restrict__ tb,
                                                   const int C, const int N,
                                                   U *__restrict__ dgamma,
                                                   U *__restrict__ dbeta)
{
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // in reduced_rows per 128
  if (tid >= C)
    return;

  U sum_dgamma = 0;
  U sum_dbeta = 0;
  for (int i = 0; i < N; i++)
  {
    U tg_curr = tg[i * C + tid];
    U tb_curr = tb[i * C + tid];
    // printf("I have %f\n",tg_curr );
    sum_dgamma += tg_curr;
    sum_dbeta += tb_curr;
  }

  dgamma[tid] = sum_dgamma;
  dbeta[tid] = sum_dbeta;
}

template <typename T, typename U>
__global__ void
InstanceNormGradBetaGamma(const T *__restrict__ dy, const T *__restrict__ x,
                          const U *__restrict__ cache_mean,
                          const U *__restrict__ cache_ivar, const int N,
                          const int C, const int D, U *__restrict__ dgamma,
                          U *__restrict__ dbeta)
{
  // Assume the total thread number == D.
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= C)
    return;

  U sum_dgamma = 0;
  U sum_dbeta = 0;
  int NxD = N * D;
  for (int i = 0; i < NxD; i++)
  {
    int in = i / D;
    int id = i % D;
    U dy_curr = GetAs<T, U>(dy, in * C * D + tid * D + id);
    sum_dgamma += dy_curr *
                  (x[in * C * D + tid * D + id] - cache_mean[in * C + tid]) *
                  cache_ivar[in * C + tid];
    sum_dbeta += dy_curr;
  }
  dgamma[tid] = sum_dgamma;
  dbeta[tid] = sum_dbeta;
}

template <typename T, typename U>
void InstanceNormGradGPU(const T *dy, const T *x, const U *cache_mean,
                         const U *cache_ivar, const U *gamma, const int N,
                         const int C, const int D, T *dx, U *dgamma, U *dbeta,
                         const U *dl_dvars_ref, const U *dl_dmus_ref,
                         const int is_channel_first)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int min_rows_per_block = 100;
  const int min_cols_per_block = 32;
  bool use_temp_space = (N * D > min_rows_per_block);

  cudaEventRecord(start);

  // if (!use_temp_space) {
  //   printf("XLOG: Dweight -> one block per column\n");
  //   printf("Run CUDA with %d blocks and each %d threads!\n",
  //          DivUp(C, kBlockSize), kBlockSize);
  //   InstanceNormGradBetaGamma<<<DivUp(C, kBlockSize), kBlockSize>>>(
  //       dy, x, cache_mean, cache_ivar, N, C, D, dgamma, dbeta, is_channel_first);
  // } else
  {
    printf("XLOG: Dweight -> multi-block per column\n");

    float *temp_dgamma;
    float *temp_dbeta;
    int total_tmp_rows;
    dim3 blocks;
    dim3 threads;
    if (is_channel_first)
    {
      const int reduced_rows =
          DivUp(N * D, min_rows_per_block * min_rows_per_block);
      const int reduced_cols = DivUp(C, min_cols_per_block);
      PrepareAlloc(&temp_dgamma, reduced_rows * C);
      PrepareAlloc(&temp_dbeta, reduced_rows * C);
      blocks.x = reduced_rows;
      blocks.y = reduced_cols;
      threads.x = kBlockSize;
      printf("XLOG: NCD, reduced_rows, reduced_cols=%d,%d\n", reduced_rows,
             reduced_cols);
      printf("XLOG: NCD, blockDIM,XYZ=%d,%d,%d\n", blocks.x, blocks.y,
             blocks.z);

      total_tmp_rows = reduced_rows;
    }
    else
    { // chanell last
      int thd_x = min(C, kBlockSize);
      int thd_y = kBlockSize / thd_x;
      int min_workload_per_thread = 3200;
      const int blocks_per_row = DivUp(N * D, thd_y * min_workload_per_thread);
      threads.x = thd_x;
      threads.y = thd_y;

      blocks.x = DivUp(C, thd_x);
      blocks.y = blocks_per_row;

      PrepareAlloc(&temp_dgamma, C * blocks_per_row * thd_y);
      PrepareAlloc(&temp_dbeta, C * blocks_per_row * thd_y);

      printf("XLOG: NDC, reduced_rows:%d, blocks dimXYZ=%d,%d,%d\n",
             blocks_per_row, blocks.x, blocks.y, blocks.z);
      printf("XLOG: NDC, threads dimXYZ=%d,%d,%d\n", threads.x, threads.y,
             threads.z);
      total_tmp_rows = min(N * D, blocks_per_row * thd_y);
    }
    InstanceNormBetaGammaRowReduceInToTemp<<<blocks, threads>>>(
        x, dy, cache_mean, cache_ivar, N, C, D, temp_dbeta, temp_dgamma,
        is_channel_first);
    InstanceNormGradBetaGammaTempToOut<<<DivUp(C, kBlockSize), kBlockSize>>>(
        temp_dgamma, temp_dbeta, C, total_tmp_rows, dgamma, dbeta);
    checkCUDA(cudaFree(temp_dgamma));
    checkCUDA(cudaFree(temp_dbeta));
  }
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds_dweight = 0;
  cudaEventElapsedTime(&milliseconds_dweight, start, stop);

  U *temp_1; // dl_dvars
  U *temp_2; // dl_dmus
  int NxC = N * C;
  PrepareAlloc(&temp_1, NxC);
  PrepareAlloc(&temp_2, NxC);

  bool use_single_warp = (D <= kWarpSize);

  const int min_num_blocks = kWarpSize;
  const int min_workload_per_thread = 50;
  bool use_single_block =
      (D <= min_num_blocks * kBlockSize * min_workload_per_thread);

  DvarOp<U, T> dl_dvar_ops{gamma, x, cache_ivar, cache_mean, C, D};
  DmeanOp<U, T> dl_dmu_ops{gamma, x, cache_ivar, cache_mean, temp_1, C, D};

  cudaEventRecord(start);
  if (use_single_warp)
  {
    printf("XLOG: Dvar/Dmean -> single-warp per row\n");
    InstanceNormRowReduceInToOutWarp<<<DivUp(NxC, kBlockSize / kWarpSize),
                                       kBlockSize>>>(
        dy, N, C, D, temp_1, temp_2, dl_dvar_ops, dl_dmu_ops, is_channel_first);
  }
  else if (use_single_block)
  {
    printf("XLOG: Dvar/Dmean -> single-block per row\n");
    if (is_channel_first)
    {
      printf("Run NCD CUDA with %d blocks and each %d threads!\n", NxC,
             kBlockSize);
      InstanceNormRowReduceInToOut<<<NxC, kBlockSize>>>(
          dy, NxC, D, temp_1, temp_2, dl_dvar_ops, dl_dmu_ops);
    }
    else
    {
      // blocks(C / 32, N, 1);
      // threads(32,1, 128/32)
      float *temp_dl_dvars;
      float *temp_dl_dmus;
      dim3 threads(kWarpSize, kBlockSize / kWarpSize);
      const int min_workload_per_thread = 3200;
      int ppr = DivUp(D, threads.y * min_workload_per_thread); //~ 10

      dim3 blocks(DivUp(C, kWarpSize), ppr, N);

      printf("XLOG: in NDC, num_blocks per row=%d\n", ppr);
      printf("XLOG: thread.XYZ=%d, %d, %d\n", threads.x, threads.y, threads.z);
      printf("XLOG: blocks.XYZ=%d, %d, %d\n", blocks.x, blocks.y, blocks.z);
      PrepareAlloc(&temp_dl_dvars, N * C * threads.y * ppr);
      PrepareAlloc(&temp_dl_dmus, N * C * threads.y * ppr);
      InstanceNormRowReduceInToOut_NDC<<<blocks, threads>>>(
          dy, N, C, D, temp_dl_dvars, dl_dvar_ops);
      InstanceNormRowReduceTempToOut<<<N * C, kBlockSize>>>(
          temp_dl_dvars, N, C, threads.y * ppr, temp_1, dl_dvar_ops);

      InstanceNormRowReduceInToOut_NDC<<<blocks, threads>>>(
          dy, N, C, D, temp_dl_dmus, dl_dmu_ops);
      InstanceNormRowReduceTempToOut<<<N * C, kBlockSize>>>(
          temp_dl_dmus, N, C, threads.y * ppr, temp_2, dl_dmu_ops);

      checkCUDA(cudaFree(temp_dl_dvars));
      checkCUDA(cudaFree(temp_dl_dmus));
    }
  }
  else
  {
    float *temp_dl_dvars;
    float *temp_dl_dmus;
    dim3 blocks, threads;
    int temp_total_rows;
    if (is_channel_first)
    {
      const int blocks_per_row = DivUp(D, kBlockSize * min_workload_per_thread);
      PrepareAlloc(&temp_dl_dvars, NxC * blocks_per_row);
      PrepareAlloc(&temp_dl_dmus, NxC * blocks_per_row);

      threads.x = kBlockSize;
      blocks.x = blocks_per_row;
      blocks.y = N;
      temp_total_rows = blocks_per_row;
      printf("XLOG: num_blocks per row=%d\n", blocks.x);
      // printf("Run CUDA with %d blocks and each %d threads!\n", NxC,
      // kBlockSize);
    }
    else
    { // channel last
      int thd_x = min(N * C, kBlockSize);
      int thd_y = kBlockSize / thd_x;
      int min_workload_per_thread = 1000;
      const int blocks_per_row = DivUp(D, thd_y * min_workload_per_thread);

      threads.x = thd_x;
      threads.y = thd_y;
      blocks.x = DivUp(N * C, thd_x);
      blocks.y = blocks_per_row;
      temp_total_rows = blocks_per_row * thd_y;
      PrepareAlloc(&temp_dl_dvars, N * C * blocks_per_row * thd_y);
      PrepareAlloc(&temp_dl_dmus, N * C * blocks_per_row * thd_y);

      printf("XLOG: Dmean/Dvar -> multi-block per row\n");
      printf("XLOG: in NDC, num_blocks per row=%d\n", blocks_per_row);
      printf("XLOG: thread.XYZ=%d, %d, %d\n", threads.x, threads.y, threads.z);
      printf("XLOG: blocks.XYZ=%d, %d, %d\n", blocks.x, blocks.y, blocks.z);
    }
    InstanceNormRowReduceInToTemp<<<blocks, threads>>>(
        dy, N, C, D, temp_dl_dvars, dl_dvar_ops, is_channel_first);
    InstanceNormRowReduceTempToOut<<<N * C, kBlockSize>>>(
        temp_dl_dvars, N, C, temp_total_rows, temp_1, dl_dvar_ops);
    InstanceNormRowReduceInToTemp<<<blocks, threads>>>(
        dy, N, C, D, temp_dl_dmus, dl_dmu_ops, is_channel_first);
    InstanceNormRowReduceTempToOut<<<N * C, kBlockSize>>>(
        temp_dl_dmus, N, C, temp_total_rows, temp_2, dl_dmu_ops);

    checkCUDA(cudaFree(temp_dl_dvars));
    checkCUDA(cudaFree(temp_dl_dmus));

    // // const int blocks_per_row = DivUp(D, kBlockSize *
    // min_workload_per_thread);

    // // float *temp_dl_dvars;
    // // float *temp_dl_dmus;
    // // PrepareAlloc(&temp_dl_dvars, NxC * blocks_per_row);
    // // PrepareAlloc(&temp_dl_dmus, NxC * blocks_per_row);

    // // dim3 threads(kBlockSize, 1, 1);
    // // dim3 blocks(blocks_per_row, N, 1);
    // // printf("XLOG: num_blocks per row=%d\n", blocks.x);
    // // // printf("Run CUDA with %d blocks and each %d threads!\n", NxC,
    // // // kBlockSize);
    // // InstanceNormRowReduceInToTemp<<<blocks, threads>>>(
    // //     dy, N, C, D, temp_dl_dvars, dl_dvar_ops);
    // // InstanceNormRowReduceTempToOut<<<NxC, threads>>>(
    // //     temp_dl_dvars, N, C, blocks_per_row, temp_1, dl_dvar_ops);

    // // InstanceNormRowReduceInToTemp<<<blocks, threads>>>(
    // //     dy, N, C, D, temp_dl_dmus, dl_dmu_ops);
    // // InstanceNormRowReduceTempToOut<<<NxC, threads>>>(
    // //     temp_dl_dmus, N, C, blocks_per_row, temp_2, dl_dmu_ops);

    // checkCUDA(cudaFree(temp_dl_dvars));
    // checkCUDA(cudaFree(temp_dl_dmus));

    // printf("XLOG: Dvar/Dmean -> multi-block per row\n");
    // const int blocks_per_row = DivUp(D, kBlockSize *
    // min_workload_per_thread);

    // float* temp_dl_dvars;
    // float* temp_dl_dmus;
    // PrepareAlloc(&temp_dl_dvars, N * blocks_per_row);
    // PrepareAlloc(&temp_dl_dmus, N * blocks_per_row);

    // dim3 threads(kBlockSize, 1, 1);
    // dim3 blocks(blocks_per_row, N, 1);
    // printf("XLOG: num_blocks per row=%d\n", blocks.x);

    // InstanceNormRowReduceInToTemp<<<blocks, threads>>>(dy, N, C, D,
    // temp_dl_dvars,
    //                                                 dl_dvar_ops);
    // InstanceNormRowReduceTempToOut<<<N, threads>>>(
    //     temp_dl_dvars, N, C, blocks_per_row, temp_1, dl_dvar_ops);

    // InstanceNormRowReduceInToTemp<<<blocks, threads>>>(dy, N,C, D,
    // temp_dl_dmus,
    //                                                 dl_dmu_ops);
    // InstanceNormRowReduceTempToOut<<<N, threads>>>(temp_dl_dmus, N, C,
    // blocks_per_row,
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
  // DxOp<T, U> dx_ops{x, cache_mean, cache_ivar, gamma, dl_dvars_ref, dl_dmus_ref, C, D};
////////////////////////////######################
  Print2D(temp_1, N, C, 1, "GPU dl_dvars:");
  Print2D(dl_dvars_ref, N, C, 1, "CPU dl_dvars:");
    Print2D(temp_2, N, C, 1, "GPU dl_dmus:");
  Print2D(dl_dmus_ref, N, C, 1, "CPU dl_dmus:");

///////////////////////////#######################

  int min_work_per_thread = 100;

  InstanceNormUpdate<<<DivUp(NxC * D, kBlockSize * min_work_per_thread),
                       kBlockSize>>>(dy, N * C * D, D, dx, dx_ops,
                                     is_channel_first);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds_update = 0;
  cudaEventElapsedTime(&milliseconds_update, start, stop);
  printf(
      "InstanceNormGradGPU time %.2f ms (dweight=%f, reduce=%f, update=%f)\n",
      milliseconds_dweight + milliseconds_reduce + milliseconds_update,
      milliseconds_dweight, milliseconds_reduce, milliseconds_update);

  checkCUDA(cudaFree(temp_1));
  checkCUDA(cudaFree(temp_2));
}

template <typename T, typename U, typename Op1, typename Op2>
__global__ void
InstanceNormRowReduceInToOutWarp(const T *__restrict__ in, const int N,
                                 const int C, const int D, U *out1, U *out2,
                                 Op1 op1, Op2 op2, const int is_channel_first)
{
  // cache_mean, cache_ivar, mean_ops, ivar_ops
  const int tid = threadIdx.x % kWarpSize;

  const int num_warps = kBlockSize / kWarpSize;
  typedef cub::WarpReduce<U> WarpReduce;
  typename WarpReduce::TempStorage temp_storage[num_warps];

  const int local_warp_id = threadIdx.x / kWarpSize;
  const int warp_id = blockIdx.x * num_warps + local_warp_id;
  int NxC = N * C;
  for (int k = warp_id; k < NxC; k += gridDim.x * num_warps)
  {
    U partial_sum = 0;
    for (int i = tid; i < D; i += kWarpSize)
    {
      if (is_channel_first)
      {
        partial_sum += op1.Compute(in, k, i);
      }
      else
      {
        partial_sum += op1.Compute_ndc(in, k / C, i, k % C);
      }
    }

    U sum = WarpReduce(temp_storage[local_warp_id]).Sum(partial_sum);

    sum = cub::ShuffleIndex<kWarpSize>(sum, 0, 0xffffffff);
    sum = op1.Finalize(sum);
    if (tid == 0)
    {
      out1[k] = sum;
    }

    partial_sum = 0;
    for (int i = tid; i < D; i += kWarpSize)
    {
      if (is_channel_first)
      {
        partial_sum += op2.Compute(in, k, i, sum);
      }
      else
      {
        partial_sum += op2.Compute_ndc(in, k / C, i, k % C, sum);
      }
    }

    sum = WarpReduce(temp_storage[local_warp_id]).Sum(partial_sum);

    if (tid == 0)
    {
      out2[k] = op2.Finalize(sum);
    }
  }
}

template <typename T, typename U, typename Op1, typename Op2>
__global__ void
InstanceNormRowReduceInToOut(const T *__restrict__ in, const int N, const int C,
                             const int D, U *out1, U *out2, Op1 op1, Op2 op2,
                             const bool is_channel_first = true)
{

  const int tid = threadIdx.x;

  typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ union
  {
    typename BlockReduce::TempStorage reduce;
    U broadcast[1];
  } temp_storage;

  for (int k = blockIdx.x; k < N * C; k += gridDim.x)
  {
    U partial_sum = 0;
    for (int i = tid; i < D; i += kBlockSize)
    {
      partial_sum += op1.Compute(in, k, i);
    }

    U sum = BlockReduce(temp_storage.reduce).Sum(partial_sum);

    if (tid == 0)
    {
      temp_storage.broadcast[0] = op1.Finalize(sum);
      out1[k] = op1.Finalize(sum);
    }
    __syncthreads();
    sum = temp_storage.broadcast[0];

    partial_sum = 0;
    for (int i = tid; i < D; i += kBlockSize)
    {
      partial_sum += op2.Compute(in, k, i, sum);
    }

    sum = BlockReduce(temp_storage.reduce).Sum(partial_sum);

    if (tid == 0)
    {
      out2[k] = op2.Finalize(sum);
    }
  }
}

template <typename T, typename U, typename Op>
__global__ void
InstanceNormRowReduceInToTemp(const T *__restrict__ x, const int N, const int C,
                              const int D, U *__restrict__ temp, Op op,
                              const int is_channel_first = true)
{
  if (is_channel_first)
  {
    typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;

    for (int row_idx = blockIdx.y; row_idx < N * C; row_idx += gridDim.y)
    {
      U partial_sum = 0;
      for (int i = row_offset; i < D; i += gridDim.x * blockDim.x)
      {
        partial_sum += op.Compute(x, row_idx, i);
      }
      U sum = BlockReduce(temp_storage).Sum(partial_sum);
      if (threadIdx.x == 0)
      {
        temp[row_idx * gridDim.x + blockIdx.x] = sum;
      }
    }
  }
  else
  {
    const int col_offset = threadIdx.y + blockIdx.y * blockDim.y;
    const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;
    if (row_offset >= N * C)
      return;
    U partial_sum = 0;
    for (int y_idx = col_offset; y_idx < D; y_idx += blockDim.y * gridDim.y)
    {
      partial_sum += op.Compute_ndc(x, row_offset / C, y_idx, row_offset % C);
    }
    const int bpr = gridDim.y;
    temp[row_offset * bpr * blockDim.y + col_offset] = partial_sum;
    //      if (threadIdx.y == 0 && col_offset < D){
    //        temp[row_offset *bpr+blockIdx.y] = partial_sum;
    //      }
  }
}

template <typename U, typename Op>
__global__ void InstanceNormRowReduceTempToOut(const U *__restrict__ temp,
                                               const int N, const int C,
                                               const int cols,
                                               U *__restrict__ cache, Op op)
{
  typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int k = blockIdx.x; k < N * C; k += gridDim.x)
  {
    U partial_sum = 0;
    for (int i = threadIdx.x; i < cols; i += kBlockSize)
    {
      partial_sum += temp[k * cols + i];
    }

    U sum = BlockReduce(temp_storage).Sum(partial_sum);

    if (threadIdx.x == 0)
    {
      cache[k] = op.Finalize(sum);
    }
  }
}

#define DTYPE float

int main(int argc, char **argv)
{
  /** Parameters and Knobs **/
  int N = 2;
  int C = 3;
  int D = 4;
  int allow_print = 0;
  int is_channel_first = 0;
  int do_test = 0;
  if (argc >= 3)
  {
    N = atoi(argv[1]);
    C = atoi(argv[2]);
    D = atoi(argv[3]);
    allow_print = atoi(argv[4]);
    is_channel_first = atoi(argv[5]);
    do_test = atoi(argv[6]);
  }

  DTYPE *x;
  float *gamma;
  float *beta;
  PrepareAlloc(&x, N * C * D);
  PrepareAlloc(&gamma, C);
  PrepareAlloc(&beta, C);

  DTYPE *y;
  float *cache_ivar;
  float *cache_mean;
  float *dl_dvars;
  float *dl_dmus;
  PrepareAlloc(&y, N * C * D);
  PrepareAlloc(&cache_ivar, N * C);
  PrepareAlloc(&cache_mean, N * C);
  PrepareAlloc(&dl_dvars, N * C);
  PrepareAlloc(&dl_dmus, N * C);

  const float epsilon = 0.001f;
  InstanceNormGPU(x, gamma, beta, epsilon, N, C, D, y, cache_mean, cache_ivar,
                  is_channel_first);
  if (allow_print)
  {
    //   Print2D(x, N, C, D, "GPU x:");
    //     Print2D(y, N, C, D, "GPU y:");
    Print2D(cache_mean, N, C, 1, "GPU cache_mean:");
    Print2D(cache_ivar, N, C, 1, "GPU cache_ivar:");
  }

  DTYPE *y_h = new DTYPE[N * C * D];
  float *cache_ivar_cpu = (float *)malloc(N * C * sizeof(float));
  float *cache_mean_cpu = (float *)malloc(N * C * sizeof(float));

  if (do_test)
    InstanceNormCPUHelper(x, gamma, beta, N, C, D, epsilon, y_h, cache_mean_cpu,
                          cache_ivar_cpu, is_channel_first);
  if (allow_print)
  {
    //     Print2DHost(y_h, N, C, D, "CPU y:");
  }
  IsClose2D(y, y_h, N, C, D, "y");
  delete[] y_h;
  // ---- Forward Done Here ----

  DTYPE *dy;
  PrepareAlloc(&dy, N * C * D, 1.2);

  DTYPE *dx;
  float *dgamma;
  float *dbeta;
  PrepareAlloc(&dx, N * C * D);
  PrepareAlloc(&dgamma, C);
  PrepareAlloc(&dbeta, C);

  printf("---------------------------------Starting of "
         "InstanceNormGradGPU!-------------------------------------\n");

  DTYPE *dx_h = new DTYPE[N * C * D];
  float *dgamma_h = new float[C];
  float *dbeta_h = new float[C];
  float *dl_dvars_h = (float *)malloc(N * C * sizeof(float));
  float *dl_dmus_h = (float *)malloc(N * C * sizeof(float));
  if (do_test)
    InstanceNormGradCPUHelper(dy, x, gamma, N, C, D, epsilon, dgamma_h, dbeta_h,
                              dx_h, dl_dvars_h, dl_dmus_h, is_channel_first);
  if (allow_print)
  {
    Print2DHost(dgamma_h, 1, 1, C, "CPU dgamma:");
    // Print2DHost(dbeta_h, 1, 1, C, "CPU dbeta:");
    Print2DHost(dx_h, N, C, D, "CPU dx:");
    Print2DHost(dl_dvars_h, N, C, 1, "CPU dl_dvars:");
    Print2DHost(dl_dmus_h, N, C, 1, "CPU dl_dmus:");
  }

  // use mean and ivar from CPU
  if (do_test){
    overwrite(cache_mean, cache_ivar, cache_mean_cpu, cache_ivar_cpu, N * C);
    overwrite(dl_dvars, dl_dmus, dl_dvars_h, dl_dmus_h, N * C);
  }
  InstanceNormGradGPU(dy, x, cache_mean, cache_ivar, gamma, N, C, D, dx, dgamma,
                      dbeta, dl_dvars, dl_dmus, is_channel_first);
  // InstanceNormGradGPU(dy, x, cache_mean, cache_ivar, gamma, N, C, D, dx,
  // dgamma,
  //                     dbeta, is_channel_first);
  if (allow_print)
  {
    Print2D(dgamma, 1, 1, C, "GPU dgamma:");
    // Print2D(dbeta, 1, 1, C, "GPU dbeta:");
    Print2D(dx, N, C, D, "GPU dx:");
    //    Print2D(dy, N, C, D, "GPU dy:");
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
  checkCUDA(cudaFree(dl_dvars));
  checkCUDA(cudaFree(dl_dmus));
  
  free(cache_ivar_cpu);
  free(cache_mean_cpu);
  free(dl_dmus_h);
  free(dl_dvars_h);
  printf(
      "###################################################################\n");
}
