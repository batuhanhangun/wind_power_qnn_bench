import os
# Set before test-module imports, as in the verified local ANN-Reg launcher.
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
