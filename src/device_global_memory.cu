#include "device_global_memory.h"
#include <cuda_runtime.h>

//vptree.cu global variables
double* d_points;
double* d_pointsAux;
unsigned int* d_indexes;
unsigned int* d_indexesAux;
unsigned int* d_vpSwaps;
double* d_treeMDs;
unsigned int* d_treeIDXs;
unsigned int* d_nodesOffset;
unsigned int* d_nodesLength;

//distances.cu global variables
double* d_distances;

//quick_select.cu global variables
double* d_qsAux;
unsigned int* d_f;
unsigned int* d_t;
unsigned int* d_addr;
unsigned int* d_NFs;
char* d_e;

/*Returns the smallest power of two*/
static unsigned int smallest_power_two(unsigned int n)
{
	unsigned int N = n;
	if ((N & (N - 1)) != 0) {	// fix if n is not power of 2
		N = 1;
		while (N < n)
			N <<= 1;
	}
	return N;
}

//function to initialize global memory
int global_memory_allocate(unsigned int numberOfPoints, unsigned int dimensionOfPoints, unsigned int maxParallelThreads)
{
	cudaError err;
	unsigned int fixedNoP = smallest_power_two(numberOfPoints - 1);		//quick select needs length in powers of two
	unsigned int maxNodes = smallest_power_two(numberOfPoints + 1) / 2;	//max nodes on the last level of the tree
	//vptree.cu global variables
	err = cudaMalloc(&d_points, (numberOfPoints * dimensionOfPoints) * sizeof(double));			if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_pointsAux, (numberOfPoints * dimensionOfPoints) * sizeof(double));		if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_indexes, numberOfPoints * sizeof(unsigned int));						if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_indexesAux, numberOfPoints * sizeof(unsigned int));						if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_vpSwaps, fixedNoP * sizeof(unsigned int));								if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_treeMDs, numberOfPoints * sizeof(double));								if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_treeIDXs, numberOfPoints * sizeof(unsigned int));						if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_nodesOffset, maxNodes * sizeof(unsigned int));							if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_nodesLength, maxNodes * sizeof(unsigned int));							if (err != cudaSuccess) return err;

	//distances.cu global variables
	err = cudaMalloc(&d_distances, fixedNoP * sizeof(double));									if (err != cudaSuccess) return err;

	//quick_select.cu global variables
	err = cudaMalloc(&d_qsAux, fixedNoP * sizeof(double));										if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_f, fixedNoP * sizeof(unsigned int));									if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_t, fixedNoP * sizeof(unsigned int));									if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_addr, fixedNoP * sizeof(unsigned int));									if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_NFs, maxParallelThreads * sizeof(unsigned int));						if (err != cudaSuccess) return err;
	err = cudaMalloc(&d_e, fixedNoP * sizeof(char));											if (err != cudaSuccess) return err;

	return cudaSuccess;
}

//function to free global memory
void global_memory_deallocate()
{
	cudaFree(d_points);
	cudaFree(d_pointsAux);
	cudaFree(d_indexes);
	cudaFree(d_indexesAux);
	cudaFree(d_vpSwaps);
	cudaFree(d_treeMDs);
	cudaFree(d_treeIDXs);
	cudaFree(d_nodesOffset);
	cudaFree(d_nodesLength);

	cudaFree(d_distances);

	cudaFree(d_qsAux);
	cudaFree(d_f);
	cudaFree(d_t);
	cudaFree(d_addr);
	cudaFree(d_e);
}
