#ifndef KNNRING_H
#define KNNRING_H

typedef struct knnresult {
	int *nidx;		//!< Indices (0-based) of nearest neighbors [m-by-k]
	double *ndist;	//!< Distance of nearest neighbors [m-by-k]
	int m;			//!< Number of query points [scalar]
	int k;			//!< Number of nearest neighbors [scalar]
} knnresult;

//! Compute k nearest neighbors of each point in X [n-by-d]
/*!
\param X Corpus data points [n-by-d]
\param Y Query data points [m-by-d]
\param n Number of corpus points [scalar]
\param m Number of query points [scalar]
\param d Number of dimensions [scalar]
\param k Number of neighbors [scalar]
\return The kNN result
*/
knnresult kNN(double *X, double  *Y, int n, int m, int d, int k);

//! Compute distributed all-kNN of points in X
/*!
\param X Data points [n-by-d]
\param n Number of data points [scalar]
\param d Number of dimensions [scalar]
\param k Number of neighbors [scalar]
\return The kNN result
*/
knnresult distrAllkNN(double *X, int n, int d, int k);

//////////////////////////////////
#include "vptree.h"
void node_to_element(double* treeMDs, unsigned int* treeIDXs, vptree* root, unsigned length);
__global__ void knn_search_init(double* d_X, double* d_Y, double* d_ndist, unsigned int* d_nidx, double* d_treeMDs, unsigned int* d_treeIDXs, unsigned int* d_offsetStack, unsigned int* d_lengthStack,
	double* d_parentMdStack, double* d_parentNDistStack, char* d_isInnerStack, unsigned int n, unsigned int m, unsigned int d, unsigned int k);
__global__ void find_nearest_kernel();

#endif