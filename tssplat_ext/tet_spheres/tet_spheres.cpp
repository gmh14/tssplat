#include "tet_spheres.h"

#include <pgo_c.h>

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

class pypgoInit
{
public:
  pypgoInit()
  {
    std::cout << "initializing" << std::endl;

    pgo_init();
    // code initialization
  }

  ~pypgoInit()
  {
    std::cout << "finalizing" << std::endl;
    // finalize
  }
};

int pgoSparseMatrixToCudaSparseMatrix(pgoSparseMatrixStructHandle m, CudaSparseMatrix &cuMat)
{
  int64_t nnz = pgo_sparse_matrix_get_num_entries(m);
  std::vector<int> rowIndices(nnz), colIndices(nnz);
  std::vector<double> valuesD(nnz);
  std::vector<float> valuesF(nnz);

  pgo_sparse_matrix_get_row_indices(m, rowIndices.data());
  pgo_sparse_matrix_get_col_indices(m, colIndices.data());
  pgo_sparse_matrix_get_values(m, valuesD.data());

  for (size_t i = 0; i < valuesD.size(); i++) {
    valuesF[i] = (float)valuesD[i];
  }

  // create cuda data
  cudaError_t cudaRet;
  cusparseStatus_t cuspRet;

  IF_CUDA_FAILED(cudaRet, cudaMalloc(&cuMat.rowIndices, nnz * sizeof(int)), return 1);
  IF_CUDA_FAILED(cudaRet, cudaMalloc(&cuMat.colIndices, nnz * sizeof(int)), return 1);
  IF_CUDA_FAILED(cudaRet, cudaMalloc(&cuMat.values, nnz * sizeof(float)), return 1);

  IF_CUDA_FAILED(cudaRet, cudaMemcpy(cuMat.rowIndices, rowIndices.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice), return 1);
  IF_CUDA_FAILED(cudaRet, cudaMemcpy(cuMat.colIndices, colIndices.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice), return 1);
  IF_CUDA_FAILED(cudaRet, cudaMemcpy(cuMat.values, valuesF.data(), sizeof(float) * nnz, cudaMemcpyHostToDevice), return 1);

  std::cout << cuMat.row << ',' << cuMat.col << std::endl;
  IF_CUSP_FAILED(cuspRet, cusparseCreateCoo(&cuMat.handle, cuMat.row, cuMat.col, nnz, cuMat.rowIndices, cuMat.colIndices, cuMat.values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), return 1);

  std::cout << "Done creating sparse matrix: " << cuMat.row << ',' << cuMat.col << std::endl;

  return 0;
}

int freeCudaSparseMatrix(CudaSparseMatrix &cuMat)
{
  cudaError_t cudaRet;
  cusparseStatus_t cuspRet;

  IF_CUDA_FAILED(cudaRet, cudaFree(cuMat.rowIndices), return 1);
  IF_CUDA_FAILED(cudaRet, cudaFree(cuMat.colIndices), return 1);
  IF_CUDA_FAILED(cudaRet, cudaFree(cuMat.values), return 1);

  IF_CUSP_FAILED(cuspRet, cusparseDestroySpMat(cuMat.handle), return 1);

  return 0;
}

int createDenseVector(float *xdata, CudaDenseVector &cuVec)
{
  // create cuda data
  cudaError_t cudaRet;
  cusparseStatus_t cuspRet;

  IF_CUDA_FAILED(cudaRet, cudaMalloc(&cuVec.value, cuVec.n * sizeof(float)), return 1);
  IF_CUDA_FAILED(cudaRet, cudaMemcpy(cuVec.value, xdata, cuVec.n * sizeof(float), cudaMemcpyHostToDevice), return 1);
  IF_CUSP_FAILED(cuspRet, cusparseCreateDnVec(&cuVec.handle, cuVec.n, cuVec.value, CUDA_R_32F), return 1);

  std::cout << "Done creating dense vec: " << cuVec.n << std::endl;

  return 0;
}

int freeDenseVector(CudaDenseVector &cuVec)
{
  // create cuda data
  cudaError_t cudaRet;
  cusparseStatus_t cuspRet;

  IF_CUDA_FAILED(cudaRet, cudaFree(cuVec.value), return 1);
  IF_CUSP_FAILED(cuspRet, cusparseDestroyDnVec(cuVec.handle), return 1);

  return 0;
}

TetSpheres::TetSpheres(const std::string &filename)
{
  // load tetmesh
  char *filename_non_const = const_cast<char *>(filename.c_str());
  pgoTetMeshGeoStructHandle tetMeshGeo = pgo_create_tetmeshgeo_from_file(filename_non_const);

  init(tetMeshGeo);

  pgo_destroy_tetmeshgeo(tetMeshGeo);
}

TetSpheres::TetSpheres(int nv, double *vertices, int ntet, int *tets)
{
  pgoTetMeshGeoStructHandle tetMeshGeo = pgo_create_tetmeshgeo(nv, vertices, ntet, tets);

  init(tetMeshGeo);

  pgo_destroy_tetmeshgeo(tetMeshGeo);
}

TetSpheres::~TetSpheres()
{
  freeCudaSparseMatrix(GTLTLG);
  freeCudaSparseMatrix(G);

  freeDenseVector(xVec);
  freeDenseVector(xTempVec);
  freeDenseVector(FVec);

  cusparseDestroy(cusp_handle);
}

void TetSpheres::init(pgoTetMeshGeoStructHandle tetMeshGeo)
{
  this->n = pgo_tetmeshgeo_get_num_vertices(tetMeshGeo);
  this->nele = pgo_tetmeshgeo_get_num_tets(tetMeshGeo);
  this->n3 = this->n * 3;
  this->alg = CUSPARSE_SPMV_COO_ALG2;

  // compute GTLTLG matrix
  pgoSparseMatrixStructHandle GTLTLG = pgo_create_tet_biharmonic_gradient_matrix(tetMeshGeo, 1, 0);
  pgoSparseMatrixStructHandle G = pgo_create_tet_gradient_matrix(tetMeshGeo);

  this->GTLTLG.row = n3, this->GTLTLG.col = n3;
  if (pgoSparseMatrixToCudaSparseMatrix(GTLTLG, this->GTLTLG) != 0) {
    throw std::runtime_error("GTLTLG sparse matrix creation");
  }

  this->G.row = nele * 9, this->G.col = n3;
  if (pgoSparseMatrixToCudaSparseMatrix(G, this->G) != 0) {
    throw std::runtime_error("G sparse matrix creation");
  }

  std::vector<float> xData(n3, 0.0f), FData(nele * 9, 0.0f);
  this->xVec.n = n3;
  if (createDenseVector(xData.data(), this->xVec) != 0) {
    throw std::runtime_error("xVec vec creation");
  }

  this->xTempVec.n = n3;
  if (createDenseVector(xData.data(), this->xTempVec) != 0) {
    throw std::runtime_error("xTempVec vec creation");
  }

  this->FVec.n = nele * 9;
  if (createDenseVector(FData.data(), this->FVec) != 0) {
    throw std::runtime_error("FVec vec creation");
  }

  this->FTempVec.n = nele * 9;
  if (createDenseVector(FData.data(), this->FTempVec) != 0) {
    throw std::runtime_error("FTempVec vec creation");
  }

  // init computation buffer
  cusparseStatus_t sp_ret;
  cudaError_t cu_ret;
  cublasStatus_t cb_ret;
  IF_CUSP_FAILED(sp_ret, cusparseCreate(&cusp_handle), throw std::runtime_error("cannot init cusp"));
  IF_CUBLAS_FAILED(cb_ret, cublasCreate(&cublas_handle), throw std::runtime_error("cannot init cublas"));
  // cublasSetMathMode(cublas_handle, CUBLAS_PEDANTIC_MATH);

  // smoothness_buffer
  float alpha = 1.0f, beta = 0.0f;
  size_t bufferSize = 0ull;
  IF_CUSP_FAILED(sp_ret, cusparseSpMV_bufferSize(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, this->GTLTLG.handle, this->xVec.handle, &beta, this->xTempVec.handle, CUDA_R_32F, alg, &bufferSize),
    throw std::runtime_error("cannot init spmv"));
  IF_CUDA_FAILED(cu_ret, cudaMalloc(&this->smoothness_buffer_dev, bufferSize), throw std::runtime_error("cannot malloc"));

  // gradient_buffer
  IF_CUSP_FAILED(sp_ret, cusparseSpMV_bufferSize(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, this->G.handle, this->xVec.handle, &beta, this->FVec.handle, CUDA_R_32F, alg, &bufferSize),
    throw std::runtime_error("cannot init spmv"));
  IF_CUDA_FAILED(cu_ret, cudaMalloc(&this->F_buffer_dev, bufferSize), throw std::runtime_error("cannot malloc"));

  IF_CUDA_FAILED(cu_ret, cudaMalloc(&this->tetJ_dev, this->nele * sizeof(float)), throw std::runtime_error("cannot malloc"));
}

torch::Tensor tet_spheres_smooth_barrier(torch::Tensor vertexPositions, TetSpheres &tetSpheres, float c1, float c2, int order);
torch::Tensor tet_spheres_smooth_barrier_backward(torch::Tensor gradH, torch::Tensor vertexPositions, TetSpheres &tetSpheres, float c1, float c2, int order);

torch::Tensor tet_spheres_forward(torch::Tensor input, std::shared_ptr<TetSpheres> tet_sph, float c1, float c2, int order)
{
  return tet_spheres_smooth_barrier(input, *tet_sph, c1, c2, order);
}

torch::Tensor tet_spheres_backward(torch::Tensor gradH, torch::Tensor input, std::shared_ptr<TetSpheres> tet_sph, float c1, float c2, int order)
{
  return tet_spheres_smooth_barrier_backward(gradH, input, *tet_sph, c1, c2, order);
}

torch::Tensor tet_spheres_random_x(std::shared_ptr<TetSpheres> tet_sph)
{
  return torch::rand({ tet_sph->n, 3 });
}

void tet_spheres_grad_limit(torch::Tensor grad, float s_threshold, float s);

PYBIND11_MODULE(tet_spheres_ext, m)
{
  static pypgoInit init;

  using pyArrayFloat = py::array_t<float, py::array::c_style | py::array::forcecast>;
  using pyArrayInt = py::array_t<int, py::array::c_style | py::array::forcecast>;

  py::class_<TetSpheres, std::shared_ptr<TetSpheres>>(m, "TetSpheres")
    .def(py::init<const std::string &>())
    .def(py::init([](pyArrayFloat vertices, pyArrayInt elements) {
      py::buffer_info vtxInfo = vertices.request();
      py::buffer_info tetInfo = elements.request();

      if (vtxInfo.ndim != (py::ssize_t)1 || vtxInfo.format != py::format_descriptor<float>::format()) {
        std::cerr << "Wrong vertex type:" << vtxInfo.ndim << ',' << vtxInfo.format << std::endl;
        return std::make_shared<TetSpheres>();
      }

#if defined(_WIN32)
      if (tetInfo.ndim != (py::ssize_t)1) {
#else
      if (tetInfo.ndim != (py::ssize_t)1 || (tetInfo.format != py::format_descriptor<int>::format())) {
#endif
        std::cerr << "Wrong tet type:" << tetInfo.ndim << ',' << tetInfo.format << ',' << py::format_descriptor<int64_t>::format() << std::endl;
        return std::make_shared<TetSpheres>();
      }

      std::vector<double> vertexPosD(vtxInfo.size);
      for (size_t i = 0; i < vertexPosD.size(); i++) {
        vertexPosD[i] = ((float *)vtxInfo.ptr)[i];
      }

      return std::make_shared<TetSpheres>((int)vertexPosD.size() / 3, vertexPosD.data(), (int)tetInfo.size / 4, (int *)tetInfo.ptr);
    }));
  // m.def("forward_cuda", &lltm_cuda_forward, "LLTM forward (CUDA)");
  // m.def("backward_cuda", &lltm_cuda_backward, "LLTM backward (CUDA)");

  m.def("forward", &tet_spheres_forward, "tet spheres forward");
  m.def("backward", &tet_spheres_backward, "tet spheres backward");
  m.def("random_x", &tet_spheres_random_x, "tet sphere random x");
  m.def("grad_limit", &tet_spheres_grad_limit, "limit_grad");
}