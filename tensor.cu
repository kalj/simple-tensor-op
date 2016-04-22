/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 *
 */

#include "utils.h"
#include "cuda_utils.cuh"
#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

#define NSUBDIV (1<<5)
#define ELEM_DEGREE 4
#define DIM 3
#define N_ITERATIONS 100

typedef float number;

int error_check()
{


    cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if( err != cudaSuccess)
  {
    printf("\n Error: %s \n Line: %d \n In file: %s", cudaGetErrorString(err), __LINE__, __FILE__);
    fflush(stdout);
    return -1;
  } 

  return 0; 

}


void swap(number *&a, number *&b)
{
  number *tmp = a;
  a = b;
  b = tmp;
}


__constant__ number phi[ELEM_DEGREE+1][ELEM_DEGREE+1];

enum Direction { X, Y, Z};
enum Transpose { TR, NOTR};

template <Direction dir, Transpose tr, unsigned int n>
__device__ void reduce(number uloc[n][n][n], number my_phi[n])
{
  // Here's what this function does for the case when dir==X and tr==TR:

  // tmp = 0;
  //
  // for(int i = 0; i < n; ++i) tmp += uloc[i][threadIdx.y][threadIdx.z]*phi[threadIdx.x][i];
  //
  // __syncthreads();
  //
  // uloc[threadIdx.x][threadIdx.y][threadIdx.z] = tmp;
  //

  // Then, the dir and tr parameters just change which index of uloc is
  // reduced, and whether phi should be transposed or not, respectively.

  number tmp = 0;

  const unsigned int reduction_idx = dir==X ? threadIdx.x : dir==Y ? threadIdx.y : threadIdx.z;

  for(int i = 0; i < n; ++i) {

    const unsigned int xidx=(dir==X) ? i : threadIdx.x;
    const unsigned int yidx=(dir==Y) ? i : threadIdx.y;
    const unsigned int zidx=(dir==Z) ? i : threadIdx.z;
    const unsigned int phi_idx1 = (tr==TR) ? i : reduction_idx;
    const unsigned int phi_idx2 = (tr==TR) ? reduction_idx : i;
    tmp += uloc[xidx][yidx][zidx]* phi[phi_idx1][phi_idx2];
  }

  __syncthreads();

  uloc[threadIdx.x][threadIdx.y][threadIdx.z] = tmp;

}


template<class T>
__device__ bool TypeIsFloat()
{
  return false;
}
template<>
__device__ bool TypeIsFloat<float>()
{
  return true;
}



template <unsigned int n>
__device__  void reduce_X(number &my_uloc, number my_phi[n], int x, int y)
{
  
  number tmp = 0.0f;
  #pragma unroll  
  for(int i = 0; i < n; ++i)
    tmp += __shfl(my_uloc, i + y*5)*my_phi[i];//uloc[i  + y*n + slice*n*n]*my_phi[i];
    // update:
    my_uloc = tmp;
}
template <unsigned int n>
__device__ __forceinline__ void reduce_Y(number &my_uloc, number my_phi[n], int x, int y)
{

  number tmp = 0.0f;
  #pragma unroll  
  for(int i = 0; i < n; ++i)
    tmp += __shfl(my_uloc, y + i*5)*my_phi[i];//uloc[i  + y*n + slice*n*n]*my_phi[i];
    // update:
    my_uloc = tmp;
}

template <unsigned int n>
__device__ __forceinline__ void reduce_Z(number &my_uloc, number my_phi[n], int x, int y)
{

   number tmp = 0.0f;
  #pragma unroll  
  for(int i = 0; i < n; ++i)
    tmp += __shfl(my_uloc, i + y*5)*my_phi[i];//uloc[i  + y*n + slice*n*n]*my_phi[i];
    // update:
    my_uloc = tmp;
}



/*
  We configure the block to have one warp work on a 5x5 "slice" at a time with a total of 5 warps in the block
  Data is staged in and out of shared memory into registers.
  When the data is in registers we use the more efficient warp shuffle to communicate data both in
  x & y direction of the 5x5 slice.

*/
template <unsigned int n, int DIM_X, int NbWarps>
__global__ void kernel(number *dst, const number *src, const unsigned int *loc2glob, const number *coeff, int CELL_PITCH)
{
  const unsigned int cell = blockIdx.x;

  __shared__ number uloc[n*n*n];
  ///////////////////////////////////////////////////////////////
  // Stage PHI in registers:
  number my_phi[n];

  // Setup one 5x5 slice per warp
  int x = threadIdx.x%5;
  int y = (threadIdx.x/5)%5;
  int slice = threadIdx.y;

  // Stage PHI untransposed
  #pragma unroll
  for(int i = 0; i < n; i++) my_phi[i] = phi[threadIdx.x%5][i];
  ///////////////////////////////////////////////////////////////
  //---------------------------------------------------------------------------
  // Phase 1: read data from global array into shared memory
  //---------------------------------------------------------------------------
  const int j = threadIdx.x + threadIdx.y*DIM_X;
  // Load global index once 
  int global_index = loc2glob [ j + cell*CELL_PITCH];

  if( j < n*n*n)
    uloc[ j ] = src[ global_index ];

  __syncthreads();

  number my_uloc = uloc[x + y*5 + slice*5*5];
  //---------------------------------------------------------------------------
  // Phase 2a-c: Interpolate -- reduce in each coordinate direction
  //---------------------------------------------------------------------------
  
  __syncthreads();
  
  // reduce in x-dir
  reduce_X<n>(my_uloc, my_phi,x,y);
  // reduce in y-dir
  reduce_Y<n>(my_uloc, my_phi,x,y);
  __syncthreads();
  // Write back to shared
  uloc[x + y*5 + slice*5*5] = my_uloc;
  __syncthreads();
  // read back to registesters in transposed fashion:
  // each 5x5 slice is now directed in the z direction.
  my_uloc = uloc[x*5*5 + y + slice*5];

  reduce_Z<n>(my_uloc, my_phi,x,y);
  __syncthreads();
  // reinsert into shared memory, transposed (this could be optimized)
   uloc[x*5*5 + y + slice*5] = my_uloc;
  __syncthreads();

  //---------------------------------------------------------------------------
  // Phase 3: apply local operations -- O(n*n*n)
  //---------------------------------------------------------------------------
  if( j < 5*5*5)
    uloc[j ] *= coeff[j + cell*CELL_PITCH];

  ///////////////////////////////////////////////////////////////
  // Stage PHI in registers: - transposed
  #pragma unroll
  for(int i = 0; i < n; i++) my_phi[i] = phi[i][threadIdx.x%5];
  //---------------------------------------------------------------------------
  // Phase 4a-c: Integrate  -- reduce with transpose
  //---------------------------------------------------------------------------
  my_uloc = uloc[x + y*5 + slice*5*5];
    __syncthreads();
  reduce_X<n>(my_uloc, my_phi , x, y);
  reduce_Y<n>(my_uloc, my_phi , x, y);
  __syncthreads();
  uloc[x + y*5 + slice*5*5] = my_uloc;
  __syncthreads();
  // read back to registers in transposed fashion:
  my_uloc = uloc[x*5*5 + y + slice*5];
  reduce_Z<n>(my_uloc, my_phi,x,y);
  __syncthreads();
  uloc[x*5*5 + y + slice*5] = my_uloc;
  __syncthreads();

  //---------------------------------------------------------------------------
  // Phase 5: write back to result
  //---------------------------------------------------------------------------
  // here, we get race conditions, but in the original code, we would launch
  // this kernel N_color times, where each launch would only work on elements
  // that are not neighbors with each other, and hence wouldn't share any data.
  if( j < 5*5*5)
  {
    // Atomic add - fire and forget!
    if( TypeIsFloat<number>() )
      atomicAdd( &dst[ global_index ], my_uloc);
    else
      dst[ global_index ] +=  uloc[j];
  }
}

int main(int argc, char *argv[])
{

  const unsigned int n_dofs = ipowf((NSUBDIV)*ELEM_DEGREE+1,DIM);
  const unsigned int n_elems = ipowf(NSUBDIV,DIM);
  const unsigned int elem_size = ipowf(ELEM_DEGREE+1,DIM);
  const unsigned int n_local_pts = n_elems*elem_size;


  unsigned int *loc2glob_cpu = new unsigned int[n_local_pts];
  number *coeff_cpu = new number[n_local_pts];

  unsigned int elemcoord[DIM];
  unsigned int dofcoord[DIM];

  printf("Elements:\t\t%d\n",n_elems);
  printf("Degrees of freedom:\t%d\n",n_dofs);

  //---------------------------------------------------------------------------
  // setup
  //---------------------------------------------------------------------------

  for(int e = 0; e < n_elems; ++e) {
    unsigned int a = 1;
    for(int d = 0; d < DIM; ++d) {
      elemcoord[d] = (e /a)% NSUBDIV;
      a *= NSUBDIV;
    }



    int i;
    for(i = 0; i < elem_size; ++i) {
      unsigned int b = 1;
      for(int d = 0; d < DIM; ++d) {
        dofcoord[d] = (i /b)% (ELEM_DEGREE+1);
        b *= (ELEM_DEGREE+1);

      }

      const unsigned int n_dofs_1d = ELEM_DEGREE*NSUBDIV+1;

      unsigned int iglob = 0;
      for(int d = 0; d < DIM; ++d) {
        iglob = iglob*n_dofs_1d + dofcoord[DIM-1-d] + elemcoord[DIM-1-d]*ELEM_DEGREE;
      }

      loc2glob_cpu[e*elem_size+i] = iglob;
      coeff_cpu[e*elem_size+i] = 1.2;
    }

  }

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();

  // Added pitched memory allocation for better memory access patterns
  // (*_*)
  const int n = ELEM_DEGREE+1;
  unsigned int *loc2glob;
  size_t widthInBytes = n*n*n*sizeof(unsigned int);
  size_t NumberOfCells = n_local_pts/(n*n*n);
  size_t c_pitch_bytes = 0;
  int c_pitch = 0;

  cudaMallocPitch( &loc2glob, &c_pitch_bytes, widthInBytes, NumberOfCells);
  // element pitch:
  c_pitch = c_pitch_bytes/sizeof(unsigned int);


  if( error_check() == -1) return -1;

  // 2D memcpy
  cudaMemcpy2D (    loc2glob, 
                    c_pitch_bytes, 
                    loc2glob_cpu,             // SRC
                    widthInBytes,            // src pitch
                    widthInBytes,             // transfer column width 
                    NumberOfCells,          // transfer height
                    cudaMemcpyHostToDevice);

  if( error_check() == -1) return -1;

  number *coeff;
  CUDA_CHECK_SUCCESS(cudaMalloc(&coeff,n_local_pts*sizeof(number)));
  CUDA_CHECK_SUCCESS(cudaMemcpy(coeff,coeff_cpu,n_local_pts*sizeof(number),
                                cudaMemcpyHostToDevice));

  number *src;
  number *dst;

  CUDA_CHECK_SUCCESS(cudaMalloc(&src,n_dofs*sizeof(number)));
  CUDA_CHECK_SUCCESS(cudaMalloc(&dst,n_dofs*sizeof(number)));

  number *cpu_arr = new number[n_dofs];
  for(int i = 0; i < n_dofs; ++i) {
    cpu_arr[i] = 1;
  }

  CUDA_CHECK_SUCCESS(cudaMemcpy(src,cpu_arr,n_dofs*sizeof(number),
                                cudaMemcpyHostToDevice));


  number phi_cpu[5*5];

  for(int i = 0; i < 5; ++i) {
    for(int j = 0; j < 5; ++j) {
      phi_cpu[i+j*5] = 0.33;
    }
  }

  CUDA_CHECK_SUCCESS(cudaMemcpyToSymbol(phi, phi_cpu, sizeof(number) * 5*5));

  //---------------------------------------------------------------------------
  // Loop
  //---------------------------------------------------------------------------


  // (*_*)
  // Setup kernel arguments, we now have a number of warps addressed by threadIdx.y in each block,
  // Every such warp will cover one 5*5*5 cell
  int CELL_PITCH = c_pitch;
  const int DIM_X = 32;
  const int NbWarps = 5;
  // Block dimensions and grid size:
  int NbBlocks = NumberOfCells;
  dim3 block(DIM_X, NbWarps);
 
  cudaDeviceSynchronize();
  double t = Timer();

  for(int i = 0; i < N_ITERATIONS; ++i)
  {
    CUDA_CHECK_SUCCESS(cudaMemset(dst, 0, n_dofs*sizeof(number)));

   kernel<n, DIM_X, NbWarps><<<NbBlocks, block>>> (dst,src,loc2glob,coeff, CELL_PITCH);
    swap(dst,src);
  }

  cudaDeviceSynchronize();
  t = Timer() -t;
  swap(dst,src);

  CUDA_CHECK_SUCCESS(cudaMemcpy(cpu_arr,dst,n_dofs*sizeof(number),
                                cudaMemcpyDeviceToHost));


  printf("Time: %8.4g s (%d iterations)\n",t,N_ITERATIONS);
  printf("Per iteration: %8.4g s\n",t/N_ITERATIONS);

  CUDA_CHECK_SUCCESS(cudaFree(loc2glob));
  CUDA_CHECK_SUCCESS(cudaFree(coeff));
  CUDA_CHECK_SUCCESS(cudaFree(dst));
  CUDA_CHECK_SUCCESS(cudaFree(src));

  delete[] cpu_arr;
  delete[] loc2glob_cpu;
  delete[] coeff_cpu;

  return 0;
}
