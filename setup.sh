# For Linux
export MKL_THREADING_LAYER=GNU

# For MacOS
export MKL_THREADING_LAYER=TBB

conda install -y clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64

MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir install torch torchvision torchaudio

python -c "import torch; print(torch.__version__)"

MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+${cpu}.html

MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+${cpu}.html

MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-geometric