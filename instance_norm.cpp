#include <cmath>
#include <iostream>

// NCD and NDC
typedef enum  {channel_first = 0, channel_last} LayOut;

template <typename T, typename U>
U GetAs(const T* in, int offset) {
  return static_cast<U>(in[offset]);
}

template <typename T, typename U>
void InstanceNormCPU(const T* x, const U* gamma, const U* beta, const int N, const int C,
                  const int D, const U epsilon, T* y, U* cache_mean_cpu, U* cache_ivar_cpu,
                  const int is_channel_first) {
  LayOut layout = is_channel_first ? channel_first : channel_last;

  int NxC = N*C;
  for (int j = 0; j < NxC; j++) {
    int iweights = j % C;
    U gamma_ch = gamma[iweights];
    U beta_ch = beta[iweights];
    U mean, ivar;
    U sum = 0;
    int jn, jc;
    U curr;
    for (int i = 0; i < D; i++) {
      if (layout == channel_last) {
        jn = j / C; jc = j % C;
        curr = GetAs<T, U>(x, jn *C *D + i*C +jc);
      } else{
        curr = GetAs<T, U>(x, j * D + i);
      }
      sum += curr;
    }
    mean = sum / D;
    cache_mean_cpu[j] = mean;
  //  printf("cpp mean=:%10.8f",mean);
    U sum_ivar = 0;
    for (int i = 0; i < D; i++) {
      if (layout == channel_last) {
        jn = j / C; jc = j % C;
        curr = GetAs<T, U>(x, jn *C *D + i*C +jc);
      } else{
        curr = GetAs<T, U>(x, j * D + i);
      }
      sum_ivar += (curr - mean) * (curr - mean);
    }
    ivar = 1.0 / sqrt(sum_ivar / D + epsilon);
    cache_ivar_cpu[j] = ivar;
  //  printf("cpp ivar=:%10.8f\n",ivar);
    for (int i = 0; i < D; i++) {
      U curr;
     // mean = 0; 
     // ivar = 1; 
     // gamma_ch = 1; beta_ch = 0;
      if (layout == channel_last) {
        jn = j / C; jc = j % C;
        int idx =  jn *C *D + i*C +jc;
        curr = GetAs<T, U>(x, idx);
        y[idx] = static_cast<T>((curr - mean) * ivar * gamma_ch + beta_ch);
      } else{
        curr = GetAs<T, U>(x, j * D + i);
        y[j * D + i] = static_cast<T>((curr - mean) * ivar * gamma_ch + beta_ch);
      }
    }
  }
}

template <typename T, typename U>
void InstanceNormGradCPU(const T* dy, const T* x, const U* gamma, const int N, const int C,
                      const int D, const U epsilon, U* dgamma, U* dbeta,
                      T* dx, const int is_channel_first) {
  LayOut layout = is_channel_first ? channel_first : channel_last;
  int NxC = N * C;
  int CxD = C * D;
  U* cache_mean = new U[NxC];
  U* cache_ivar = new U[NxC];
  int jn,jc;
  for (int j = 0; j < NxC; j++) {
    U mean, ivar;
    U sum = 0;
    for (int i = 0; i < D; i++) {
      int idx;
      if (layout == channel_last) {
        jn = j / C; jc = j % C;
        idx = jn *C *D + i*C +jc;
      } else{
        idx = j * D + i;  
      }
      U curr =  GetAs<T, U>(x, idx);
      sum += curr;
    }
    mean = sum / D;

    U sum_ivar = 0;
    for (int i = 0; i < D; i++) {
      int idx;
      if (layout == channel_last) {
        jn = j / C; jc = j % C;
        idx = jn *C *D + i*C +jc;
      } else {
        idx = j * D + i;
      }
      U curr = GetAs<T, U>(x, idx);
      sum_ivar += (curr - mean) * (curr - mean);
    }
    ivar = 1.0 / sqrt(sum_ivar / D + epsilon);

    cache_mean[j] = mean;
    cache_ivar[j] = ivar;
   // printf("cpp mean=:%10.8f\n",mean);
   // printf("cpp ivar=:%10.8f\n",ivar);
  }

  // Compute dgamma, dbeta.
  for (int i = 0; i < C; i++) {
    dgamma[i] = 0;
    dbeta[i] = 0;
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < D; k++){
        int idx;
        if (layout == channel_last) {
          idx = j * CxD + k * C + i;
        } else {
          idx = j * CxD + i * D + k;
        }
        U dy_curr = static_cast<U>(dy[idx]);
        dgamma[i] += dy_curr* (x[idx] - cache_mean[ j * C + i]) * cache_ivar[ j * C + i];
        dbeta[i] +=  dy_curr;
      }
    }
  }

  // Compute dx.
  for (int i = 0; i < NxC; i++) {
    U dl_dvar = 0;
    int in = i / C;
    int ic = i % C;
    for (int j = 0; j < D; j++) {
      int idx = (layout == channel_last) ? in * C*D+j*C+ic : i*D+j;
      U curr = static_cast<U>(dy[idx]);
      dl_dvar += curr * gamma[ic] * (x[idx] - cache_mean[i]) * (-0.5) *
                 (cache_ivar[i] * cache_ivar[i] * cache_ivar[i]);
    }

    U dl_dmean = 0;
    for (int j = 0; j < D; j++) {
      int idx = (layout == channel_last) ? in * C*D+j*C+ic : i*D+j;
      U curr = static_cast<U>(dy[idx]);
      dl_dmean += -1. * curr * gamma[ic] * cache_ivar[i];
      dl_dmean += dl_dvar * (-2. / D) * (x[idx] - cache_mean[i]);
    }

    for (int j = 0; j < D; j++) {
      int idx = (layout == channel_last) ? in * C*D+j*C+ic : i*D+j;
      U curr = static_cast<U>(dy[idx]);
      U dl_di = curr * gamma[ic] * cache_ivar[i];
      U di_dx = 1.;

      // dl_dvar is above.
      U dvar_dx = 2. * (x[idx] - cache_mean[i]) / D;

      // dl_dmean is above.
      U dmean_dx = 1. / D;

      U dl_dx = dl_di * di_dx + dl_dvar * dvar_dx + dl_dmean * dmean_dx;
      dx[idx] = static_cast<T>(dl_dx);
    }
  }

  delete[] cache_mean;
  delete[] cache_ivar;
}

template <typename T>
void IsClose2DHost(const T* x, const T* y, int N, int C, int D, std::string msg,
                   float atol = 1e-3, float rtol = 1e-3) {
  bool is_same = true;
  int NxC = N *C;
  for (int i = 0; i < NxC; i++) {
    for (int j = 0; j < D; j++) {
      float d_val = static_cast<float>(x[j + i * D]);
      float h_val = static_cast<float>(y[j + i * D]);
      if (fabs(d_val - h_val) > (atol + rtol * fabs(h_val))) {
        is_same = false;
        printf("Found diff: CPU=%f, GPU=%f at (%d, %d)\n", h_val, d_val, i, j);
        break;
      }
    }
    if (!is_same) break;
  }
  printf("Test (%s): %s\n", msg.c_str(), is_same ? "True" : "False");
}

template <typename T>
void Print2DHost(const T* x, int N, int C, int D, std::string msg) {
  printf("%s\n", msg.c_str());
  for (int i = 0; i < N * C; i++) {
    for (int j = 0; j < D; j++) {
      printf("%f, ", static_cast<float>(x[j + i * D]));
    }
    printf("\n");
  }
}

extern "C" {
void instance_norm(const float* x, const float* gamma, const float* beta,
                const int N, const int C, const int D, const float epsilon,
                 float* y, float *mean, float* ivar,  const int is_channel_first) {
  InstanceNormCPU(x, gamma, beta, N, C, D, epsilon, y, mean, ivar, is_channel_first);
}

void instance_norm_grad(const float* dy, const float* x, const float* gamma,
                     const int N, const int C, const int D, const float epsilon, float* dx,
                     float* dgamma, float* dbeta, const int is_channel_first) {
  InstanceNormGradCPU(dy, x, gamma, N, C, D, epsilon, dgamma, dbeta, dx, is_channel_first);
}

void is_close_2d_host(const float* x, const float* y, int N, int C, int D,
                      std::string msg, float atol = 1e-3, float rtol = 1e-3) {
  IsClose2DHost(x, y, N, C, D, msg, atol, rtol);
}

void print_2d(const float* x, int N, int C, int D, std::string msg) {
  Print2DHost(x, N, C, D, msg);
}
}



