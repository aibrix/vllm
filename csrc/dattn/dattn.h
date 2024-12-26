#pragma once
/*
 Copyright (c) ByteDance Inc.
 Authors: 
  - Tongping Liu (tongping.liu@bytedance.com)
  - https://github.com/vllm-project/vllm/pull/6102/commits
 */ 
//#include <torch/script.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cstddef>
#include <deque>
#include <unordered_map>
#include <torch/custom_class.h>
#include <c10/util/intrusive_ptr.h>
#include <pthread.h>


#define _MB (1 << 20)

using namespace std;

static inline void
checkRtError(cudaError_t res, const char *tok, const char *file, unsigned line) {
    if (res != cudaSuccess) {
        std::cerr << file << ':' << line << ' ' << tok
                  << "failed (" << (unsigned)res << "): " << cudaGetErrorString(res) << std::endl;
        abort();
    }
}

#define CHECK_RT(x) checkRtError(x, #x, __FILE__, __LINE__);

static inline void
checkDrvError(CUresult res, const char *tok, const char *file, unsigned line) {
    if (res != CUDA_SUCCESS) {
        const char *errStr = NULL;
        (void)cuGetErrorString(res, &errStr);
        std::cerr << file << ':' << line << ' ' << tok
                  << "failed (" << (unsigned)res << "): " << errStr << std::endl;
        abort();
    }
}

#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);


// kvCacheRegion class, to warp CUdeviceptr, used to kv-cache tensor
// record the reserved virtual address size and allocated physical memory size.
// TODO: we may avoid expose this class externally in the future. 
class kvCacheRegion : public torch::CustomClassHolder{
private:
  // the starting address of this region
  char * dptr;

  // the size of a kv cache block in bytes
  uint64_t cache_block_size; 

  // the size of a physical block, which is the multiple of cache_block_size and is aligned with page_size
  uint64_t physical_block_size; 

  // The actual size that has been mapped successfully
  uint64_t mapped_size; 

  // virtual address of the next page that needs to be mapped. 
  // Typically, (nextUnmappedAddr - dptr)/page_size == total_pagees 
  char * nextUnmappedAddr; 

public:

  kvCacheRegion(int64_t cache_block_size, int64_t physical_block_size, CUdeviceptr ptr);

  ~kvCacheRegion();

  // get the number of physical pages
  CUdeviceptr getStartPtr(void); 
  
  void updateBlocks(uint64_t blocks, cudaStream_t stream);
  void freeAllPhyMemory(void);
};


// kvCacheAllocator class, used for memory allocation of kv-cachemanager, memory allocation is based on page granularity,
class kvCacheAllocator : public torch::CustomClassHolder{
private:
  CUcontext torchContext;  
  CUcontext origContext;   

  int64_t region_size; 
  int64_t cache_block_size;
  int64_t physical_block_size; 
  uint64_t page_size;
  CUdevice device;
  std::mutex mutex;

  cudaStream_t stream;
  
  // the hashtable to record the relationship between regions and ptrs
  std::unordered_map<uint64_t, kvCacheRegion*> active_regions_map;

  // Internal functions
  static void *memoryManagerThread(void * arg); 
  // Release the virtual address space for a region that is related to one request
  void _releaseRegion(int64_t region_id);

  bool manager_running;
  bool immediate_allocate; 
  
  pthread_t thread_id;
  pthread_mutex_t mutex_manager;
  pthread_cond_t  cond_manager; 
  std::vector<int64_t> free_caches;
  std::vector<std::vector<int64_t>> req_cache_blocks; 
  std::vector<std::vector<int64_t>> swap_out_caches; 
  std::vector<std::vector<int64_t>> swap_in_caches; 

public:

  // The default contructor. Otherwise, torch bindings will complain it. 
  kvCacheAllocator(int64_t max_gpu_memory_size, int64_t cache_block_size, int64_t region_cache_size);


  ~kvCacheAllocator() = default;

   // get the granularity of the physical memory allocation
  int64_t getPageSize(void);

  // Reserve the virtual address space for a region that is related to one request
  // In particular, the regionSize == 2 * max_seq_length * layers_num * heads_num * head_size * dtype_size
  // "2" here is to allocate Key and Value cache together, which helps to reduce the fragmentation 
  int64_t reserveRegion(int64_t region_id);
  std::vector<int64_t> allocCPUCaches(int64_t num_caches, int64_t cache_size);
  void releaseRegions(std::vector<int64_t> regions);

  void allocCacheBlocks(std::vector<std::vector<int64_t>> reqs_blocks, cudaStream_t stream);

  void updateCacheBlocks(bool immediate_allocate, std::vector<int64_t> free_caches, std::vector<std::vector<int64_t>> req_caches, 
                      std::vector<std::vector<int64_t>> to_swap_out, std::vector<std::vector<int64_t>> to_swap_in);

  void swapOutCache(std::vector<std::vector<int64_t>> swap_caches, cudaStream_t stream); 
  void swapInCache(std::vector<std::vector<int64_t>> swap_caches, cudaStream_t stream); 

};

