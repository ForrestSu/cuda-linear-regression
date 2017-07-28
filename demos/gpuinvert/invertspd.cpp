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
 * invert.cpp
 * Copyright (C) 2009 Remco Bouckaert
 * remco@cs.waikato.ac.nz, rrb@xm.co.nz
 */
 
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include "config.h"

// When VERIFY is defined, the sum of squared errors is calculated between the
// identity matrix and the product A * incerse(A). For debugging...
//#define VERIFY 1

// size of panels can impact performance, try some values
#define PANEL_SIZE 32



/* implementatin of Cholesky decomposition on the CPU */
void choleskyDecompositionCPU(int n, REAL * A, int lda) {
	REAL * L = new REAL[n*n];
	memset(L,0,sizeof(REAL)*n*n);

	for (int j = 0; j < n; j++) {
		REAL d = 0.0;
		for (int k = 0; k < j; k++) {
			REAL s = 0.0;
			for (int i = 0; i < k; i++) {
				s += L[k*n+i] * L[j*n+i];
			}
			L[j*n+k] = s = (A[j+k*lda] - s) / L[k*n+k];
			L[k*n+j] = L[j*n+k];
			d = d + s * s;
		}
		d = A[j*(lda+1)] - d;
		L[j*n+j] = (d > 0 ? sqrt(d) : 0);
	}
	for(int j = 0;j < n; j++) {
		for (int k = 0; k < n; k++) {
			A[j+k*lda] = L[j*n+k];
		}
	}
	delete [] L;
	return;
} // choleskyDecompositionCPU

void solveCPU ( const int m, const int n,
        const REAL *A, const int lda, REAL *B,
             const int ldb) {
  int i, j, k;
  int n1, n2;
    for (i = 0; i < n; i++) {
        REAL Aii = A[lda * i + i];
        for (j = 0; j < m; j++) {
          B[ldb * i + j] /= Aii;
	 }

      for (k = i + 1; k < n; k++) {
        const REAL Aik = A[i * lda + k];
        for (j = 0; j < m; j++) {
          B[ldb * k + j] -= Aik * B[ldb * i + j];
        }
      }
    }
} // solveCPU
             	
/* implementation of Cholesky decomposition on the GPU 
 * following Vasily Volkov and James Demmel. For technical details,
 * see "LU, QR and Cholesky Factorizations using Vector Capabilities of GPUs"
 * May 13, 2008
 * http://www.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-49.pdf
 */
void choleskyDecompositionGPU( int n, REAL *cpu_A, REAL * A, int lda)
{	
    const int BLOCK_SIZE = 128;
    SAFECALL(cudaMemcpy(A, cpu_A, n*lda*sizeof(REAL),cudaMemcpyHostToDevice ) );
    
    //  iterate through block columns
    for( int i = 0; i < n; i += PANEL_SIZE ) {
        int h = n - i;
        int w = h < PANEL_SIZE ? h : PANEL_SIZE;
        if( i > 0 ) {
#ifdef DOUBLE_PRECISION	
		  cublasDsyrk( 'L', 'N', w, PANEL_SIZE, -1, 
		  	&A[i+(i-PANEL_SIZE)*lda], lda, 1, 
		  	&A[i*(lda+1)], lda );
#else	  	  
		  cublasSsyrk( 'L', 'N', w, PANEL_SIZE, -1, 
		  	&A[i+(i-PANEL_SIZE)*lda], lda, 1, 
		  	&A[i*(lda+1)], lda );
#endif		  	
	  	  SAFECALL( cublasGetError( ) );

#ifdef DOUBLE_PRECISION	
	  	  cublasDgemm( 'N', 'T', h-w, w, PANEL_SIZE, -1, 
	  	  	&A[i+PANEL_SIZE+(i-PANEL_SIZE)*lda], lda, 
	  	  	&A[i+(i-PANEL_SIZE)*lda], lda, 1, 
	  	  	&A[i+PANEL_SIZE+i*lda], lda );
#else	  	  
	  	  cublasSgemm( 'N', 'T', h-w, w, PANEL_SIZE, -1, 
	  	  	&A[i+PANEL_SIZE+(i-PANEL_SIZE)*lda], lda, 
	  	  	&A[i+(i-PANEL_SIZE)*lda], lda, 1, 
	  	  	&A[i+PANEL_SIZE+i*lda], lda );
#endif	  	  	
	  	  SAFECALL( cublasGetError( ) );

	  	  SAFECALL( cudaMemcpy2D( &cpu_A[i*(lda+1)], lda*sizeof(REAL), 
	  	  	&A[i*(lda+1)], lda*sizeof(REAL), h*sizeof(REAL), w, cudaMemcpyDeviceToHost ) );
            
          if( h > PANEL_SIZE ) {
#ifdef DOUBLE_PRECISION	
	      	cublasDsyrk( 'L', 'N', h-PANEL_SIZE, PANEL_SIZE, -1, 
	      		&A[i+PANEL_SIZE+(i-PANEL_SIZE)*lda], lda, 1, 
	      		&A[(i+PANEL_SIZE)*(lda+1)], lda );
#else
	      	cublasSsyrk( 'L', 'N', h-PANEL_SIZE, PANEL_SIZE, -1, 
	      		&A[i+PANEL_SIZE+(i-PANEL_SIZE)*lda], lda, 1, 
	      		&A[(i+PANEL_SIZE)*(lda+1)], lda );
#endif	
	 	    SAFECALL( cublasGetError( ) );
	      }
        }
        
        choleskyDecompositionCPU(w, &cpu_A[i*(lda+1)], lda);
        
        if( h > PANEL_SIZE ) {
	  		solveCPU(h - PANEL_SIZE, PANEL_SIZE, 
	  			&cpu_A[i*(lda+1)], lda, 
	  			&cpu_A[i+PANEL_SIZE+i*lda], lda );
	  		SAFECALL( cudaMemcpy2D( &A[i*(lda+1)], lda*sizeof(REAL), 
	  			&cpu_A[i*(lda+1)], lda*sizeof(REAL), h*sizeof(REAL), w, cudaMemcpyHostToDevice ) );
        }
    }
} // choleskyDecompositionGPU	


/* Inverts the nxn symmetric positive definite (SPD) matrix A stored with 
 * column length lda.
 * The result is stored back in A.
 * There is no attempt made to verify that A is SPD.
 */
void invert(REAL * A, int lda, int n) {
  fprintf(stderr,"inversion started");
#ifdef VERIFY	
	REAL * xcopy =new REAL[n*n];
	  for (int i = 0; i < n; i++) {
	    for (int j = 0; j < n; j++) {
	      xcopy[i*n+j] = A[i*lda+j];
	    }
	  }
	  // mathdisp(xcopy,n,n);
#endif

    volatile clock_t gputime;
    gputime=clock();

    REAL * A_d;
    int m = (n+31)&~31;
    SAFECALL(cudaMalloc((void**) &A_d, lda*m*2*sizeof(REAL)));
    choleskyDecompositionGPU(n, A,  A_d, lda);
	SAFECALL(cudaMemcpy(A_d, A, lda*n*sizeof(REAL),cudaMemcpyHostToDevice));
        
	// create identity matrix
	REAL *B_h = new REAL[n*lda];
	memset(B_h, 0, lda*n*sizeof(REAL));
	for (int i=0;i<n;i++){
	  B_h[i*lda+i] = 1;
	}

	// solve
	REAL *B = &A_d[lda*m];
	SAFECALL(cudaMemcpy(B, B_h, lda*n*sizeof(REAL), cudaMemcpyHostToDevice));
#ifdef DOUBLE_PRECISION	
	cublasDtrsm ('L', 'L', 'N', 'N', n, n, 1.0f, A_d, lda, B, lda);
#else	
	cublasStrsm ('L', 'L', 'N', 'N', n, n, 1.0f, A_d, lda, B, lda);
#endif	
	cudaThreadSynchronize();

	// solve
#ifdef DOUBLE_PRECISION	
	cublasDtrsm ('L', 'L', 'T', 'N', n, n, 1.0f, A_d, lda, B, lda);
#else	
	cublasStrsm ('L', 'L', 'T', 'N', n, n, 1.0f, A_d, lda, B, lda);
#endif	
	cudaThreadSynchronize();
	SAFECALL(cudaMemcpy(A, B, lda*n*sizeof(REAL), cudaMemcpyDeviceToHost));

	SAFECALL(cudaFree(A_d));
	delete [] B_h;

	// now A = inverse(spd_matrix)
	gputime=clock()-gputime;fprintf(stderr, " %7.2f ms ",gputime/1.e3f);

    fprintf(stderr, " %7.2f Gflops", 1e-3*(2.0+3.0+3.0)*n*n*n/3.0/gputime);
#ifdef VERIFY	
	// let's verify that
	REAL error=0.0;

	// multiply inverse*xcopy, should be Identity matrix
	for (int k = 0; k < n; k++) {
	  for (int j = 0; j < n; j++) {
	    REAL sum = 0;
	    for (int i = 0; i < n; i++) {
	      sum += A[j*lda+i]*xcopy[i*n+k];
	    }
	    if (j!=k) {
	      error += sum * sum;
	    } else {
	      error += (1.0-sum) * (1.0-sum);
	    }
	  }
	}
        fprintf(stderr, " %6.6f SSE", error);
#endif	
	fprintf(stderr," done!\n");
} // invert 
