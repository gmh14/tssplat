pip install pybind11
python -m pip install scikit-build
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# install pypgo
if [ -z "${CUDA_HOME}" ]; then
  echo "CUDA_HOME is not defined!"
  exit 1
else
  echo "CUDA_HOME is defined as ${CUDA_HOME}"
fi
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

pip install trimesh xatlas omegaconf opencv-python matplotlib tqdm mitsuba pymeshlab
pip install --upgrade PyMCubes
pip install ninja

python -m pip install gurobipy

pip install git+https://github.com/NVlabs/nvdiffrast.git

# tinycudann
git clone --recursive git@github.com:NVlabs/tiny-cuda-nn.git
cd tiny-cuda-nn/bindings/torch
python setup.py install

# for custom compiled gcc, may also need to manually replace the libstdc++.so.6 inside conda env with the one in the gcc lib path
