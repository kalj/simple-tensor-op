/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 *
 */


#include "utils.h"
#include "cuda_utils.cuh"
#include "timer.h"


#define NSUBDIV (1<<5)
#define ELEM_DEGREE 4
#define DIM 3
#define N_ITERATIONS 100

#define ROW_LENGTH 128

typedef double number;

void swap(number *&a, number *&b)
{
    number *tmp = a;
    a = b;
    b = tmp;
}


__constant__ double phi[ELEM_DEGREE+1][ELEM_DEGREE+1];

enum Direction { X, Y, Z};
enum Transpose { TR, NOTR};

    template <Direction dir, Transpose tr, unsigned int n>
__device__ void reduce(number uloc[n][n][n])
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

    for(int i = 0; i < n; ++i)
        tmp += (uloc[(dir==X) ? i : threadIdx.x][(dir==Y) ? i : threadIdx.y][(dir==Z) ? i : threadIdx.z])*
            phi[(tr==TR) ? reduction_idx : i][(tr==TR) ? i : reduction_idx];

    __syncthreads();

    uloc[threadIdx.x][threadIdx.y][threadIdx.z] = tmp;

}

    template <unsigned int n>
__global__ void kernel(number *dst, const number *src, const unsigned int *loc2glob, const number *coeff)
{
    const unsigned int cell = blockIdx.x;
    const unsigned int tid = threadIdx.x+n*threadIdx.y + n*n*threadIdx.z;

    __shared__ number uloc[n][n][n];

    // read data from global array into shared memory
    uloc[threadIdx.x][threadIdx.y][threadIdx.z] = src[loc2glob[cell*n*n*n+tid]];
    __syncthreads();

    // reduce along x -- O(n*n*n*n)
    reduce<X,TR,n> (uloc);
    __syncthreads();

    // reduce along y
    reduce<Y,TR,n> (uloc);
    __syncthreads();

    // reduce along z
    reduce<Z,TR,n> (uloc);

    // now we should have values at quadrature points
    // no synch is necessary since we are only working on local data.

    // apply local operations -- O(n*n*n)
    uloc[threadIdx.x][threadIdx.y][threadIdx.z] *= coeff[cell*n*n*n+tid];

    __syncthreads();

    // now integrate

    // reduce along x
    reduce<X,NOTR,n> (uloc);
    __syncthreads();

    // reduce along y
    reduce<Y,NOTR,n> (uloc);
    __syncthreads();

    // reduce along z
    reduce<Z,NOTR,n> (uloc);
    __syncthreads();

    // write back to result

    // here, we get race conditions, but in the original code, we would launch
    // this kernel N_color times, where each launch would only work on elements
    // that are not neighbors with each other, and hence wouldn't share any data.

    dst[loc2glob[cell*n*n*n+tid]] += uloc[threadIdx.x][threadIdx.y][threadIdx.z];
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

        // fill with crap
        for(;i<ROW_LENGTH;i++) {
            loc2glob_cpu[e*elem_size+i] = 0;
            coeff_cpu[e*elem_size+i] = -1;
        }


    }

    unsigned int *loc2glob;
    CUDA_CHECK_SUCCESS(cudaMalloc(&loc2glob,n_local_pts*sizeof(unsigned int)));
    CUDA_CHECK_SUCCESS(cudaMemcpy(loc2glob,loc2glob_cpu,n_local_pts*sizeof(unsigned int),
                cudaMemcpyHostToDevice));


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

    dim3 bk_dim(ELEM_DEGREE+1,ELEM_DEGREE+1,ELEM_DEGREE+1);
    dim3 gd_dim(n_elems);

    cudaDeviceSynchronize();
    double t = Timer();

    for(int i = 0; i < N_ITERATIONS; ++i)
    {
        CUDA_CHECK_SUCCESS(cudaMemset(dst, 0, n_dofs*sizeof(number)));

        // kernel<ELEM_DEGREE+1> <<<gd_dim,bk_dim>>> (dst,src,loc2glob);
        kernel<ELEM_DEGREE+1> <<<gd_dim,bk_dim>>> (dst,src,loc2glob,coeff);
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
