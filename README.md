# LBM_OpenACC
Use OpenACC to accelerate Lattice Boltzmann Method (LBM, D3Q15)

The source file shows how to use OpenACC to accelerate LBM, and includes a CUDA implementation of LBM to compare the performance.

Compilation

Don't use OpenACC acceleration
    g++ 3D_LBM_Poiseuille.cpp -O2 -o without_acc
  
Use OpenACC acceleration
    pgc++ -O2 -acc -Minfo=accel -ta=tesla 3D_LBM_Poiseuille.cpp -o with_acc
Use CUDA acceleration: comment line 17 and uncomment line 17, and rename the file with suffix .cu.
    nvcc -arch=sm_35 3D_LBM_Poiseuille.cpp -o with_cuda
    
Performance

OpenACC : CPU = 68X (2.7s VS 181.5s)
CUDA    : OpenACC = 1.91X (1.4s VS 2.7s)

Platform

CPU: i7-4930 CPU @ 3.40GHz (only one core is used in the test)
GPU: Tesla K40c
