#pragma once

#include "cudaUtils.h"

#include <pgo_c.h>

#include <string>

struct CudaSparseMatrix
{
  int row, col;
  int *rowIndices = nullptr, *colIndices = nullptr;
  float *values = nullptr;
  cusparseSpMatDescr_t handle;
};

struct CudaDenseVector
{
  int n;
  float *value = nullptr;
  cusparseDnVecDescr_t handle;
};

struct TetSpheres
{
  TetSpheres(const std::string &filename);
  TetSpheres(int nv, double *vertices, int ntet, int *tets);
  TetSpheres() {}
  ~TetSpheres();

  void init(pgoTetMeshGeoStructHandle tetMeshGeo);

  cusparseHandle_t cusp_handle;
  cublasHandle_t cublas_handle;

  CudaSparseMatrix GTLTLG, G;
  CudaDenseVector xVec, xTempVec, FVec, FTempVec;
  cusparseSpMVAlg_t alg;
  float *tetF_dev = nullptr, *tetJ_dev = nullptr;
  float *smoothness_buffer_dev = nullptr, *F_buffer_dev = nullptr, *F_buffer2_dev = nullptr;
  int n3 = 0, n = 0, nele = 0;
};
