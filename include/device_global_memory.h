#ifndef DEVICE_GLOBAL_MEMORY_H_
#define DEVICE_GLOBAL_MEMORY_H_

/*This header has all pointers to device global memory from the host*/

//vptree.cu global variables
extern double* d_points;
extern double* d_pointsAux;
extern unsigned int* d_indexes;
extern unsigned int* d_indexesAux;
extern unsigned int* d_vpSwaps;
extern double* d_treeMDs;
extern unsigned int* d_treeIDXs;
extern unsigned int* d_nodesOffset;
extern unsigned int* d_nodesLength;

//distances.cu global variables
extern double* d_distances;

//quick_select.cu global variables
extern double* d_qsAux;
extern unsigned int* d_f;
extern unsigned int* d_t;
extern unsigned int* d_addr;
extern unsigned int* d_NFs;
extern char* d_e;

//function to initialize global memory
int global_memory_allocate(unsigned int numberOfPoints, unsigned int dimensionOfPoints, unsigned int maxParallelThreads);

//function to free global memory
void global_memory_deallocate();

#endif // !DEVICE_GLOBAL_MEMORY_H_
