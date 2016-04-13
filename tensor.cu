/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 *
 */

#include "utils.h"
#include "cuda_utils.cuh"
#include "timer.h"
#include "atomic.cuh"

#include <vector>

#define NSUBDIV (1<<5)
#define ELEM_DEGREE 4
#define DIM 3
#define N_ITERATIONS 100
#define ROWLENGTH 128
#define NUMCOLORS 8

typedef double number;

void swap(number *&a, number *&b)
{
  number *tmp = a;
  a = b;
  b = tmp;
}


__constant__ number phi[25];
__constant__ number dphi[25];

enum Direction { X, Y, Z};
enum Transpose { TR, NOTR};

template <Direction dir, unsigned int n,
            bool add, bool inplace>
__device__ void reduce(number *dst, const number *src, const number myphi[n])
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

  // Then, the dir and tr parameters just change which index of uloc is reduced,
  // and whether phi should be transposed or not, respectively. with_gradients
  // decide wether to reduce with phi or dphi. add decide wether to add to the
  // result, and inplace decides whether src and dst are infact the same vector.

  number tmp = 0;

#pragma unroll
  for(int i = 0; i < n; ++i) {

    const unsigned int srcidx =
        (dir==X) ? (i + n*(threadIdx.y + n*threadIdx.z))
        : (dir==Y) ? (threadIdx.z + n*(i + n*threadIdx.y))
        : (threadIdx.y + n*(threadIdx.z + n*i));

      tmp += myphi[i] * (inplace ? dst[srcidx] : src[srcidx]);
  }

  if(inplace) __syncthreads();

  const unsigned int dstidx =
      (dir==X) ? (threadIdx.x + n*(threadIdx.y + n*threadIdx.z))
      : (dir==Y) ? (threadIdx.z + n*(threadIdx.x + n*threadIdx.y))
      : (threadIdx.y + n*(threadIdx.z + n*threadIdx.x));

  if(add)
    dst[dstidx] += tmp;
  else
    dst[dstidx] = tmp;
}


template <unsigned int n>
__global__ void kernel_grad(number *__restrict__ dst, const number *__restrict__ src,
                            const unsigned int *__restrict__ loc2glob,
                            const number *__restrict__ coeff, const number *__restrict__ jac,
                            const number *__restrict__ jxw)
{
  const unsigned int nqpts=n*n*n;
  const unsigned int cell = blockIdx.x;
  const unsigned int ncells = gridDim.x;
  const unsigned int tid = threadIdx.x+n*threadIdx.y + n*n*threadIdx.z;

  __shared__ number values[nqpts];
  __shared__ number gradients[3][nqpts];

  ///////////////////////////////////////////////////////////////
  // Stage PHI and DPHI in registers:
  number my_phi[n];
  number my_dphi[n];
#pragma unroll
  for(int i = 0; i < n; i++) {
    my_phi[i] = phi[threadIdx.x + n*i];
    my_dphi[i] = dphi[threadIdx.x + n*i];
  }
  ///////////////////////////////////////////////////////////////

  //---------------------------------------------------------------------------
  // Phase 1: read data from global array into shared memory
  //---------------------------------------------------------------------------
  values[tid] = __ldg(&src[loc2glob[cell*ROWLENGTH+tid]]);
  __syncthreads();

  //---------------------------------------------------------------------------
  // Phase 2a-c: Interpolate -- reduce in each coordinate direction
  //---------------------------------------------------------------------------


  // reduce along x / i / q - direction
  reduce<X,n,false,false> (gradients[0],values,my_dphi);
  reduce<X,n,false,false> (gradients[1],values,my_phi);
  reduce<X,n,false,false> (gradients[2],values,my_phi);
  __syncthreads();

  // reduce along y / j / r - direction
  reduce<Y,n,false,true> (gradients[0],gradients[0],my_phi);
  reduce<Y,n,false,true>  (gradients[1],gradients[1],my_dphi);
  reduce<Y,n,false,true> (gradients[2],gradients[2],my_phi);
  __syncthreads();

  // reduce along z / k / s - direction
  reduce<Z,n,false,true> (gradients[0],gradients[0],my_phi);
  reduce<Z,n,false,true> (gradients[1],gradients[1],my_phi);
  reduce<Z,n,false,true>  (gradients[2],gradients[2],my_dphi);

  __syncthreads();
  // now we should have values at quadrature points
  // no synch is necessary since we are only working on local data.

  //---------------------------------------------------------------------------
  // Phase 3: apply local operations -- O(n*n*n)
  //---------------------------------------------------------------------------

  number grad[DIM];
#pragma unroll
  for(int d1=0; d1<DIM; d1++) {
    number tmp = 0;
#pragma unroll
    for(int d2=0; d2<DIM; d2++) {
      tmp += jac[((DIM*d2+d1)*ncells+cell)*ROWLENGTH+tid]*gradients[d2][tid];
    }
    grad[d1] = tmp;
  }

  grad[0] *= coeff[cell*ROWLENGTH+tid];
  grad[1] *= coeff[cell*ROWLENGTH+tid];
  grad[2] *= coeff[cell*ROWLENGTH+tid];

#pragma unroll
  for(int d1=0; d1<DIM; d1++) {
    number tmp = 0;
#pragma unroll
    for(int d2=0; d2<DIM; d2++) {
      tmp += jac[((DIM*d1+d2)*ncells+cell)*ROWLENGTH+tid]*grad[d2];
    }
    gradients[d1][tid] = tmp*jxw[cell*ROWLENGTH+tid];
  }


  __syncthreads();
  ///////////////////////////////////////////////////////////////
  // Stage transpose of PHI and DPHI in registers:
#pragma unroll
  for(int i = 0; i < n; i++) {
    my_phi[i] = phi[threadIdx.x*n + i];
    my_dphi[i] = dphi[threadIdx.x*n + i];
  }
  ///////////////////////////////////////////////////////////////

  __syncthreads();


  //---------------------------------------------------------------------------
  // Phase 4a-c: Integrate  -- reduce with transpose
  //---------------------------------------------------------------------------

  reduce<X,n,false,true>  (gradients[0],gradients[0],my_dphi);
  reduce<X,n,false,true> (gradients[1],gradients[1],my_phi);
  reduce<X,n,false,true> (gradients[2],gradients[2],my_phi);
  __syncthreads();

  // reduce along y / j / r - direction
  reduce<Y,n,false,true> (gradients[0],gradients[0],my_phi);
  reduce<Y,n,false,true>  (gradients[1],gradients[1],my_dphi);
  reduce<Y,n,false,true> (gradients[2],gradients[2],my_phi);
  __syncthreads();

  // reduce along z / k / s - direction
  reduce<Z,n,false,false> (values,gradients[0],my_phi);
  __syncthreads();
  reduce<Z,n,true,false> (values,gradients[1],my_phi);
  __syncthreads();
  reduce<Z,n,true,false> (values,gradients[2],my_dphi);

  __syncthreads();
  // no synch is necessary since we are only working on local data.

  //---------------------------------------------------------------------------
  // Phase 5: write back to result
  //---------------------------------------------------------------------------

  // here, we get race conditions, but in the original code, we would launch
  // this kernel N_color times, where each launch would only work on elements
  // that are not neighbors with each other, and hence wouldn't share any data.

#if NUMCOLORS == 1
  atomicAddWrapper(&dst[loc2glob[cell*ROWLENGTH+tid]],values[tid]);
#else
  dst[loc2glob[cell*ROWLENGTH+tid]] += values[tid];
#endif
}


template <unsigned int n>
__global__ void kernel(number *__restrict__ dst, const number *__restrict__ src,
                       const unsigned int *__restrict__ loc2glob,
                       const number *__restrict__ coeff,const number *__restrict__ jxw)
{
  const unsigned int nqpts=n*n*n;
  const unsigned int cell = blockIdx.x;
  const unsigned int tid = threadIdx.x+n*threadIdx.y + n*n*threadIdx.z;

  __shared__ number values[nqpts];

  ///////////////////////////////////////////////////////////////
  // Stage PHI in registers:
  number my_phi[n];
#pragma unroll
  for(int i = 0; i < n; i++) {
    my_phi[i] = phi[threadIdx.x*n+i];
  }
  ///////////////////////////////////////////////////////////////

  //---------------------------------------------------------------------------
  // Phase 1: read data from global array into shared memory
  //---------------------------------------------------------------------------
  values[tid] = __ldg(&src[loc2glob[cell*ROWLENGTH+tid]]);
  __syncthreads();

  //---------------------------------------------------------------------------
  // Phase 2a-c: Interpolate -- reduce in each coordinate direction
  //---------------------------------------------------------------------------


  // reduce along x -- O(n*n*n*n)
  reduce<X,n,false,true> (values,values,my_phi);
  __syncthreads();

  // reduce along y
  reduce<Y,n,false,true> (values,values,my_phi);
  __syncthreads();

  // reduce along z
  reduce<Z,n,false,true> (values,values,my_phi);

  // now we should have values at quadrature points
  // no synch is necessary since we are only working on local data.

  //---------------------------------------------------------------------------
  // Phase 3: apply local operations -- O(n*n*n)
  //---------------------------------------------------------------------------

  values[tid] *= coeff[cell*ROWLENGTH+tid]*jxw[cell*ROWLENGTH+tid];

  __syncthreads();

  ///////////////////////////////////////////////////////////////
  // Stage PHI in registers: - transpose
#pragma unroll
  for(int i = 0; i < n; i++) {
    my_phi[i] = phi[threadIdx.x+n*i];
  }
  ///////////////////////////////////////////////////////////////
  __syncthreads();

  //---------------------------------------------------------------------------
  // Phase 4a-c: Integrate  -- reduce with transpose
  //---------------------------------------------------------------------------

  // reduce along x
  reduce<X,n,false,true> (values,values,my_phi);
  __syncthreads();

  // reduce along y
  reduce<Y,n,false,true> (values,values,my_phi);
  __syncthreads();

  // reduce along z
  reduce<Z,n,false,true> (values,values,my_phi);

  // __syncthreads();
  // no synch is necessary since we are only working on local data.

  //---------------------------------------------------------------------------
  // Phase 5: write back to result
  //---------------------------------------------------------------------------

  // here, we get race conditions, but in the original code, we would launch
  // this kernel N_color times, where each launch would only work on elements
  // that are not neighbors with each other, and hence wouldn't share any data.

#if NUMCOLORS == 1
  atomicAddWrapper(&dst[loc2glob[cell*ROWLENGTH+tid]],values[tid]);
#else
  dst[loc2glob[cell*ROWLENGTH+tid]] += values[tid];
#endif
}



int main(int argc, char *argv[])
{
  const unsigned int n_dofs = ipowf((NSUBDIV)*ELEM_DEGREE+1,DIM);
  const unsigned int n_elems = ipowf(NSUBDIV,DIM);
  const unsigned int elem_size = ipowf(ELEM_DEGREE+1,DIM);
  const unsigned int n_local_pts = n_elems*ROWLENGTH;

  std::vector<unsigned int> n_cells(NUMCOLORS);
  for(int c=0; c<NUMCOLORS; ++c)
    n_cells[c] = 0;

  std::vector<std::vector<unsigned int>> loc2glob_cpu(NUMCOLORS);
  std::vector<std::vector<number>> coeff_cpu(NUMCOLORS);
  std::vector<std::vector<number>> jxw_cpu(NUMCOLORS);
  std::vector<std::vector<number>> jac_cpu(NUMCOLORS);


  unsigned int elemcoord[DIM];
  unsigned int dofcoord[DIM];

  printf("Elements:\t\t%d\n",n_elems);
  printf("Degrees of freedom:\t%d\n",n_dofs);

  //---------------------------------------------------------------------------
  // setup
  //---------------------------------------------------------------------------

  std::vector<unsigned int> loc2glob_tmp(ROWLENGTH);
  std::vector<number> coeff_tmp(ROWLENGTH);
  std::vector<number> jac_tmp(DIM*DIM*ROWLENGTH);
  std::vector<number> jxw_tmp(ROWLENGTH);

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

      loc2glob_tmp[i] = iglob;
      coeff_tmp[i] = 1.2;

      jxw_tmp[i] = 0.493;
      for(int d=0; d<(DIM*DIM); ++d) {
        jac_tmp[d*elem_size+i] = 1.1;
      }

    }

    unsigned int color = 0;
    if(NUMCOLORS > 1) {
      color = elemcoord[0] % 2 + 2*(elemcoord[1]%2) + 4*(elemcoord[2]%2);
    }

    loc2glob_cpu[color].insert(loc2glob_cpu[color].end(),
                                loc2glob_tmp.begin(),
                                loc2glob_tmp.end());

    coeff_cpu[color].insert(coeff_cpu[color].end(),
                             coeff_tmp.begin(),
                             coeff_tmp.end());

    jxw_cpu[color].insert(jxw_cpu[color].end(),
                           jxw_tmp.begin(),
                           jxw_tmp.end());

    jac_cpu[color].insert(jac_cpu[color].end(),
                           jac_tmp.begin(),
                           jac_tmp.end());
    n_cells[color]++;

  }

  std::vector<unsigned int *> loc2glob(NUMCOLORS);
  std::vector<number *> coeff(NUMCOLORS);
  std::vector<number *> jac(NUMCOLORS);
  std::vector<number *> jxw(NUMCOLORS);

  for(int c=0; c<NUMCOLORS; ++c) {
    {
      unsigned int size = ROWLENGTH*n_cells[c]*sizeof(unsigned int);
      CUDA_CHECK_SUCCESS(cudaMalloc(&loc2glob[c],size));
      CUDA_CHECK_SUCCESS(cudaMemcpy(loc2glob[c],loc2glob_cpu[c].data(),
                                    size, cudaMemcpyHostToDevice));
    }

    {
      unsigned int size = ROWLENGTH*n_cells[c]*sizeof(number);
      CUDA_CHECK_SUCCESS(cudaMalloc(&coeff[c],size));
      CUDA_CHECK_SUCCESS(cudaMemcpy(coeff[c],coeff_cpu[c].data(),
                                    size, cudaMemcpyHostToDevice));
    }

    {
      unsigned int size = ROWLENGTH*n_cells[c]*DIM*DIM*sizeof(number);
      CUDA_CHECK_SUCCESS(cudaMalloc(&jac[c],size));
      CUDA_CHECK_SUCCESS(cudaMemcpy(jac[c],jac_cpu[c].data(),
                                    size, cudaMemcpyHostToDevice));
    }

    {
      unsigned int size = ROWLENGTH*n_cells[c]*sizeof(number);
      CUDA_CHECK_SUCCESS(cudaMalloc(&jxw[c],size));
      CUDA_CHECK_SUCCESS(cudaMemcpy(jxw[c],jxw_cpu[c].data(),
                                    size, cudaMemcpyHostToDevice));
    }

  }

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
  CUDA_CHECK_SUCCESS(cudaMemcpyToSymbol(dphi, phi_cpu, sizeof(number) * 5*5));

  //---------------------------------------------------------------------------
  // Loop
  //---------------------------------------------------------------------------
  printf("Setup done\n");

  if(sizeof(number) == 8) {
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  }

  dim3 bk_dim(ELEM_DEGREE+1,ELEM_DEGREE+1,ELEM_DEGREE+1);
  dim3 gd_dim[NUMCOLORS];

  for(int c=0; c<NUMCOLORS; ++c)
    gd_dim[c] = dim3(n_cells[c]);

  cudaDeviceSynchronize();
  double t = Timer();

  for(int i = 0; i < N_ITERATIONS; ++i)
  {
    CUDA_CHECK_SUCCESS(cudaMemset(dst, 0, n_dofs*sizeof(number)));

    // kernel<ELEM_DEGREE+1> <<<gd_dim,bk_dim>>> (dst,src,loc2glob,coeff,jxw);
    for(int c=0; c<NUMCOLORS; ++c) {
      kernel_grad<ELEM_DEGREE+1> <<<gd_dim[c],bk_dim>>> (dst,src,loc2glob[c],coeff[c],
                                                         jac[c],jxw[c]);
      CUDA_CHECK_LAST;
    }
    swap(dst,src);
  }

  cudaDeviceSynchronize();
  t = Timer() -t;
  swap(dst,src);

  CUDA_CHECK_SUCCESS(cudaMemcpy(cpu_arr,dst,n_dofs*sizeof(number),
                                cudaMemcpyDeviceToHost));


  printf("Time: %8.4g s (%d iterations)\n",t,N_ITERATIONS);
  printf("Per iteration: %8.4g s\n",t/N_ITERATIONS);

  for(int c=0; c<NUMCOLORS; ++c) {
    CUDA_CHECK_SUCCESS(cudaFree(loc2glob[c]));
    CUDA_CHECK_SUCCESS(cudaFree(coeff[c]));
    CUDA_CHECK_SUCCESS(cudaFree(jac[c]));
    CUDA_CHECK_SUCCESS(cudaFree(jxw[c]));
  }
  CUDA_CHECK_SUCCESS(cudaFree(dst));
  CUDA_CHECK_SUCCESS(cudaFree(src));

  delete[] cpu_arr;

  return 0;
}
