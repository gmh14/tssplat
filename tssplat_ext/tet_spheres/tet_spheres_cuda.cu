#include "cudaUtils.h"
#include "tet_spheres.h"

#include <torch/extension.h>

#include <iostream>
#include <iomanip>

template<typename Scalar = float>
__host__ __device__ inline const Scalar &elt(const Scalar A[9], int row, int col)
{
  return A[col * 3 + row];
}

template<typename Scalar = float>
__host__ __device__ inline Scalar &elt(Scalar A[9], int row, int col)
{
  return A[col * 3 + row];
}

template<typename Scalar = float>
__host__ __device__ Scalar det(const Scalar M[9])
{
  return -elt(M, 0, 2) * elt(M, 1, 1) * elt(M, 2, 0) +
    elt(M, 0, 1) * elt(M, 1, 2) * elt(M, 2, 0) +
    elt(M, 0, 2) * elt(M, 1, 0) * elt(M, 2, 1) -
    elt(M, 0, 0) * elt(M, 1, 2) * elt(M, 2, 1) -
    elt(M, 0, 1) * elt(M, 1, 0) * elt(M, 2, 2) +
    elt(M, 0, 0) * elt(M, 1, 1) * elt(M, 2, 2);
}

template<typename Scalar = float>
__host__ __device__ void ddetA_dA(const Scalar A[9], Scalar ddetA_dAOut[9])
{
  elt(ddetA_dAOut, 0, 0) = -(elt(A, 1, 2) * elt(A, 2, 1)) + elt(A, 1, 1) * elt(A, 2, 2);
  elt(ddetA_dAOut, 0, 1) = elt(A, 1, 2) * elt(A, 2, 0) - elt(A, 1, 0) * elt(A, 2, 2);
  elt(ddetA_dAOut, 0, 2) = -(elt(A, 1, 1) * elt(A, 2, 0)) + elt(A, 1, 0) * elt(A, 2, 1);

  elt(ddetA_dAOut, 1, 0) = elt(A, 0, 2) * elt(A, 2, 1) - elt(A, 0, 1) * elt(A, 2, 2);
  elt(ddetA_dAOut, 1, 1) = -(elt(A, 0, 2) * elt(A, 2, 0)) + elt(A, 0, 0) * elt(A, 2, 2);
  elt(ddetA_dAOut, 1, 2) = elt(A, 0, 1) * elt(A, 2, 0) - elt(A, 0, 0) * elt(A, 2, 1);

  elt(ddetA_dAOut, 2, 0) = -(elt(A, 0, 2) * elt(A, 1, 1)) + elt(A, 0, 1) * elt(A, 1, 2);
  elt(ddetA_dAOut, 2, 1) = elt(A, 0, 2) * elt(A, 1, 0) - elt(A, 0, 0) * elt(A, 1, 2);
  elt(ddetA_dAOut, 2, 2) = -(elt(A, 0, 1) * elt(A, 1, 0)) + elt(A, 0, 0) * elt(A, 1, 1);
}

__global__ void cuda_forward_det(int nele, float *tetF, float *tetJ, int order)
{
  int gidx = (blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  if (gidx >= nele) {
    return;
  }

  float J = det(tetF + gidx * 9);
  J = fmaxf(-J, 0.0f);
  float Jpow = 0.f;
  if (order == 2) {
    Jpow = J * J;
  }
  else if (order == 4) {
    Jpow = J * J * J * J;
  }

  tetJ[gidx] = Jpow;
}

__global__ void cuda_backward_det(int nele, float *tetF, float *tet_dJ_dF, int order)
{
  int gidx = (blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  if (gidx >= nele) {
    return;
  }

  float J = det(tetF + gidx * 9);

  if (J < 0) {
    J = -J;

    float dJdF[9];
    ddetA_dA(tetF + gidx * 9, dJdF);

    float Jpow = 0.f;
    if (order == 2) {
      Jpow = 2.0f * J;
    }
    else if (order == 4) {
      Jpow = 4.0f * J * J * J;
    }

#pragma unroll
    for (int i = 0; i < 9; i++) {
      tet_dJ_dF[gidx * 9 + i] = dJdF[i] * -Jpow;
    }
  }
  else {
#pragma unroll
    for (int i = 0; i < 9; i++) {
      tet_dJ_dF[gidx * 9 + i] = 0.0f;
    }
  }
}

__global__ void cuda_grad_limit(int n, float *grad, int max_idx, float s)
{
  int gidx = (blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  if (gidx >= n) {
    return;
  }

  grad[gidx] = grad[gidx] / grad[max_idx] * s;
}

auto round_up_n2 = [](int v, int d) -> int {
  return ((1 + ((v)-1) / (d)) * (d));
};

torch::Tensor tet_spheres_smooth_barrier(torch::Tensor vertexPositions, TetSpheres &tetSpheres, float c1, float c2, int order)
{
  cusparseStatus_t sp_ret;
  cudaError_t cu_ret;
  cublasStatus_t cb_ret;

  auto vertexPosFlat = vertexPositions.contiguous();
  cusparseDnVecDescr_t vertexPosFlatDnVec;
  IF_CUSP_FAILED(sp_ret, cusparseCreateDnVec(&vertexPosFlatDnVec, tetSpheres.n3, vertexPosFlat.data_ptr(), CUDA_R_32F),
    throw std::runtime_error("bind torch tensor buffer"));

  // step one x GTLTLG x * 0.5
  float alpha = 1.0f, beta = 0.0f;
  IF_CUSP_FAILED(sp_ret, cusparseSpMV(tetSpheres.cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, tetSpheres.GTLTLG.handle, vertexPosFlatDnVec, &beta, tetSpheres.xTempVec.handle, CUDA_R_32F, tetSpheres.alg, tetSpheres.smoothness_buffer_dev),
    throw std::runtime_error("GTLTLG x failed"));

  // std::vector<float> xTemp(tetSpheres.xTempVec.n), x(tetSpheres.xTempVec.n);
  // IF_CUDA_FAILED(cu_ret, cudaMemcpy(xTemp.data(), tetSpheres.xTempVec.value, sizeof(float) * xTemp.size(), cudaMemcpyDeviceToHost),
  //   throw std::runtime_error("cuda memcpy xtemp"));

  // IF_CUDA_FAILED(cu_ret, cudaMemcpy(x.data(), vertexPosFlat.data_ptr(), sizeof(float) * x.size(), cudaMemcpyDeviceToHost),
  //   throw std::runtime_error("cuda memcpy x"));

  // std::cout << "Mv:\n";
  // for (int i = 0; i < 6; i++) {
  //   std::cout << xTemp[i] << ',';
  // }
  // std::cout << std::endl;

  // std::cout << "Mx:" << xTemp[0] << ',' << xTemp[1] << ',' << xTemp[2] << std::endl;
  // std::cout << "x:" << x[0] << ',' << x[1] << ',' << x[2] << std::endl;

  // cublasHandle_t cublas_handle;
  // IF_CUBLAS_FAILED(cb_ret, cublasCreate(&cublas_handle), throw std::runtime_error("cannot init cublas"));

  float sm_energy = 0;
  IF_CUBLAS_FAILED(cb_ret, cublasSdot(tetSpheres.cublas_handle, tetSpheres.n3, tetSpheres.xTempVec.value, 1, reinterpret_cast<float *>(vertexPosFlat.data_ptr()), 1, &sm_energy),
    throw std::runtime_error("cublas dot failed"));

  sm_energy *= 0.5;

  // float sm_energy1 = 0;
  // for (int i = 0; i < (int)xTemp.size(); i++) {
  //   sm_energy1 += xTemp[i] * x[i];
  // }
  // sm_energy1 *= 0.5;
  // std::cout << std::setprecision(8) << "eng:" << sm_energy1 << ',' << sm_energy << std::endl;

  // step two G x
  IF_CUSP_FAILED(sp_ret, cusparseSpMV(tetSpheres.cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, tetSpheres.G.handle, vertexPosFlatDnVec, &beta, tetSpheres.FVec.handle, CUDA_R_32F, tetSpheres.alg, tetSpheres.F_buffer_dev),
    throw std::runtime_error("G x failed"));

  int numDetPerBlock = 1024;
  int warpSize = 32;
  int numWarpPerBlock = numDetPerBlock / warpSize;
  int numBlocks = round_up_n2(tetSpheres.nele, numDetPerBlock) / numDetPerBlock;

  dim3 dimBlock;
  dimBlock.x = warpSize;
  dimBlock.y = numWarpPerBlock;

  dim3 dimGrid;
  dimGrid.x = numBlocks;

  cuda_forward_det<<<dimGrid, dimBlock>>>(tetSpheres.nele, tetSpheres.FVec.value, tetSpheres.tetJ_dev, order);

  float barrier_energy = 0;
  IF_CUBLAS_FAILED(cb_ret, cublasSasum(tetSpheres.cublas_handle, tetSpheres.nele, tetSpheres.tetJ_dev, 1, &barrier_energy),
    throw std::runtime_error("cublas dot failed"));

  IF_CUSP_FAILED(sp_ret, cusparseDestroyDnVec(vertexPosFlatDnVec),
    throw std::runtime_error("dn vec"));

  float finalEnergy = sm_energy * c1 + barrier_energy * c2;
  //std::cout << std::setprecision(8) << "sm: " << sm_energy << "; bar: " << barrier_energy << std::endl;

  return torch::tensor(finalEnergy);
}

torch::Tensor tet_spheres_smooth_barrier_backward(torch::Tensor gradH, torch::Tensor vertexPositions, TetSpheres &tetSpheres, float c1, float c2, int order)
{
  cusparseStatus_t sp_ret;
  cudaError_t cu_ret;
  cublasStatus_t cb_ret;

  auto vertexPosFlat = vertexPositions.contiguous();
  cusparseDnVecDescr_t vertexPosFlatDnVec;
  IF_CUSP_FAILED(sp_ret, cusparseCreateDnVec(&vertexPosFlatDnVec, tetSpheres.n3, vertexPosFlat.data_ptr(), CUDA_R_32F),
    throw std::runtime_error("bind torch tensor buffer"));

  auto gradFinal = torch::zeros_like(vertexPositions).contiguous();

  cusparseDnVecDescr_t gradFinalDnVec;
  IF_CUSP_FAILED(sp_ret, cusparseCreateDnVec(&gradFinalDnVec, tetSpheres.n3, gradFinal.data_ptr(), CUDA_R_32F),
    throw std::runtime_error("bind torch tensor buffer"));

  // step one GTLTLG x
  float alpha = c1, beta = 0.0f;
  IF_CUSP_FAILED(sp_ret, cusparseSpMV(tetSpheres.cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, tetSpheres.GTLTLG.handle, vertexPosFlatDnVec, &beta, gradFinalDnVec, CUDA_R_32F, tetSpheres.alg, tetSpheres.smoothness_buffer_dev),
    throw std::runtime_error("GTLTLG x failed"));

  // step two G x
  alpha = 1.0f, beta = 0.0f;
  IF_CUSP_FAILED(sp_ret, cusparseSpMV(tetSpheres.cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, tetSpheres.G.handle, vertexPosFlatDnVec, &beta, tetSpheres.FVec.handle, CUDA_R_32F, tetSpheres.alg, tetSpheres.F_buffer_dev),
    throw std::runtime_error("G x failed"));

  // sum fmax^p(-J(Gx), 0))
  // if J < 0
  // sum p (-J)^{p - 1} -dJ/dF dF/dx
  // (1, nele) (nele, nele * 9) * (nele * 9, n3)

  int numDetPerBlock = 1024;
  int warpSize = 32;
  int numWarpPerBlock = numDetPerBlock / warpSize;
  int numBlocks = round_up_n2(tetSpheres.nele, numDetPerBlock) / numDetPerBlock;

  dim3 dimBlock;
  dimBlock.x = warpSize;
  dimBlock.y = numWarpPerBlock;

  dim3 dimGrid;
  dimGrid.x = numBlocks;

  // IF_CUDA_FAILED(cu_ret, cudaMemset(tetSpheres.FTempVec.value, 0, sizeof(float) * tetSpheres.FTempVec.n),
  //   throw std::runtime_error("memset df failed"));

  cuda_backward_det<<<dimGrid, dimBlock>>>(tetSpheres.nele, tetSpheres.FVec.value, tetSpheres.FTempVec.value, order);

  alpha = c2;
  beta = 1.0;
  IF_CUSP_FAILED(sp_ret, cusparseSpMV(tetSpheres.cusp_handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, tetSpheres.G.handle, tetSpheres.FTempVec.handle, &beta, gradFinalDnVec, CUDA_R_32F, tetSpheres.alg, tetSpheres.F_buffer2_dev),
    throw std::runtime_error("G x failed"));

  IF_CUSP_FAILED(sp_ret, cusparseDestroyDnVec(vertexPosFlatDnVec),
    throw std::runtime_error("dn vec"));

  IF_CUSP_FAILED(sp_ret, cusparseDestroyDnVec(gradFinalDnVec),
    throw std::runtime_error("dn vec"));

  float v = gradH.item<float>();
  IF_CUBLAS_FAILED(cb_ret, cublasSscal(tetSpheres.cublas_handle, tetSpheres.n3, &v, (float *)gradFinal.data_ptr(), 1),
    throw std::runtime_error("cublas scale"));
  // std::cout << gradFinal.size(0) << ',' << gradFinal.size(1) << std::endl;

  return gradFinal;
}

void tet_spheres_grad_limit(torch::Tensor grad, float s_threshold, float s)
{
  cublasStatus_t cb_ret;
  cudaError_t cu_ret;
  cublasHandle_t cublas_handle;

  int ret = 0;
  int n = grad.size(0) * grad.size(1);
  float max_val;
  std::cout << grad.size(0) << ',' << grad.size(1) << ',' << n << std::endl;

  IF_CUBLAS_FAILED(cb_ret, cublasCreate(&cublas_handle), return);
  IF_CUBLAS_FAILED(cb_ret, cublasIsamax(cublas_handle, n, (float *)grad.data_ptr(), 1, &ret), return);
  IF_CUBLAS_FAILED(cu_ret, cudaMemcpy(&max_val, (float *)grad.data_ptr(), sizeof(float), cudaMemcpyDeviceToHost), return);

  if (max_val > s_threshold) {
    float v = s / max_val;
    IF_CUBLAS_FAILED(cb_ret, cublasSscal(cublas_handle, n, &v, (float *)grad.data_ptr(), 1), return);
  }

  // int numDetPerBlock = 1024;
  // int warpSize = 32;
  // int numWarpPerBlock = numDetPerBlock / warpSize;
  // int numBlocks = round_up_n2(n, numDetPerBlock) / numDetPerBlock;

  // dim3 dimBlock;
  // dimBlock.x = warpSize;
  // dimBlock.y = numWarpPerBlock;

  // dim3 dimGrid;
  // dimGrid.x = numBlocks;

  // // IF_CUDA_FAILED(cu_ret, cudaMemset(tetSpheres.FTempVec.value, 0, sizeof(float) * tetSpheres.FTempVec.n),
  // //   throw std::runtime_error("memset df failed"));

  // cuda_grad_limit<<<dimGrid, dimBlock>>>(n, (float *)grad.data_ptr(), ret, s);

  cublasDestroy(cublas_handle);
}