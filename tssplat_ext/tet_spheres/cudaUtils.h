#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

#include <cstdio>

#define IF_CUSOLVER_FAILED(cs, prc, err_prc)                                                              \
  do {                                                                                                    \
    (cs) = (prc);                                                                                         \
    if ((cs) != CUSOLVER_STATUS_SUCCESS) {                                                                \
      fprintf(stderr, "%s failed! error code: %d at line %d, file %s\n", #prc, (cs), __LINE__, __FILE__); \
      err_prc;                                                                                            \
    }                                                                                                     \
  } while (0)

#define IF_CUBLAS_FAILED(cs, prc, err_prc)                                                                \
  do {                                                                                                    \
    (cs) = (prc);                                                                                         \
    if ((cs) != CUBLAS_STATUS_SUCCESS) {                                                                  \
      fprintf(stderr, "%s failed! error code: %d at line %d, file %s\n", #prc, (cs), __LINE__, __FILE__); \
      err_prc;                                                                                            \
    }                                                                                                     \
  } while (0)

#define IF_CUSP_FAILED(cs, prc, err_prc)                                                                  \
  do {                                                                                                    \
    (cs) = (prc);                                                                                         \
    if ((cs) != CUSPARSE_STATUS_SUCCESS) {                                                                \
      fprintf(stderr, "%s failed! error code: %d at line %d, file %s\n", #prc, (cs), __LINE__, __FILE__); \
      err_prc;                                                                                            \
    }                                                                                                     \
  } while (0)

#define IF_CUDA_FAILED(cs, prc, err_prc)                                                                  \
  do {                                                                                                    \
    (cs) = (prc);                                                                                         \
    if ((cs) != cudaSuccess) {                                                                            \
      fprintf(stderr, "%s failed! error code: %d at line %d, file %s\n", #prc, (cs), __LINE__, __FILE__); \
      err_prc;                                                                                            \
    }                                                                                                     \
  } while (0)
