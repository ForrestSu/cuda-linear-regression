/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * test.cpp
 * Copyright (C) 2009 Remco Bouckaert
 * remco@cs.waikato.ac.nz, rrb@xm.co.nz
 */

#include <stdio.h>
#include <string.h> 
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include "config.h"

#include "helper_cuda.h"

extern void invert(REAL * A, int lda, int n);

/**
 * usage:
 *  ./testspd -dev 0 -n 1024
 */
int main(int argc, char** argv)
{   
    int n=1024;
    int dev = 0;
    for( int i = 1; i < argc-1 ; i ++ ) {
        if( strcmp( argv[i], "-dev" ) == 0 ) {
           dev = atoi( argv[i+1] );
        }
        if( strcmp( argv[i], "-n" ) == 0 ) {
            n = atoi( argv[i+1] );
        }
    }
    printf("Using device %i with n=%i\n", dev, n);    
    
    
    if( cudaSetDevice( dev ) != cudaSuccess )
    {
      printf( "Failed to set device %d\n", dev );
      return 1;
    }
    
    if( cublasInit() != CUBLAS_STATUS_SUCCESS )
    {
      printf( "failed to initialize the cublas library\n" );
      return 1;
    }
    printf("Cublas initialized...\n");

    REAL *A;
    int lda = ( ((n+15)&(~15)) |16); //min is 16, increase step by 32

    checkCudaErrors( cudaMallocHost((void**)&A, n*lda*sizeof(REAL)) );


    srand(n);
    for( int i = 0; i < n; i++ ) {
    	for (int j = i; j < n; j++) {
    		A[i*lda+j] = 2.0*(rand()%32768)/32768.0;
        	A[j*lda+i] = A[i*lda+j];
    	} 
	A[i*lda+i] += 2*sqrt((REAL)n);
    }


    invert(A, lda, n);

    checkCudaErrors(cudaFreeHost( A ));

    return 0;
} // main
