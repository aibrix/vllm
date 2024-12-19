/*
 Copyright (c) ByteDance Inc.
 Authors: 
  - Tongping Liu (tongping.liu@bytedance.com)
 */ 
 
#include <c10/core/ScalarType.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <cuda_runtime.h>
#include <Python.h>
#include <pthread.h>

#include "dattn.h"


#define KV_UTILIZATION_RATE (0.9)

static CUmemAllocationProp _prop = {};
static CUmemAccessDesc _accessDescr = {};
 
/* 
  In this allocator, we only have the following concepts, but without the concept of tokens.
  The python portion should convert the number of tokens to tokens depending on their block_size (e.g., 16)
  Region: virtual address space for a request. Currently, we support the space for max_seq_len.
 */
static uint64_t roundup(uint64_t size, uint64_t align_size) {
  return ((size + align_size - 1)/align_size) * align_size; 
}

static int allocatePhyPages(void * ptr, uint64_t size) {
  CUdeviceptr dptr = (CUdeviceptr)ptr;

  CUdevice dev; // device
  CHECK_DRV(cuCtxGetDevice(&dev));
  _prop.location.id = dev;
  _accessDescr.location = _prop.location;

  CUresult status = CUDA_SUCCESS;
  CUmemGenericAllocationHandle allocationHandle;
  if ((status = cuMemCreate(&allocationHandle, size, &_prop, 0)) == CUDA_SUCCESS) {
    if ((status = cuMemMap(dptr, size, 0ULL, allocationHandle, 0ULL)) == CUDA_SUCCESS) {
      if ((status = cuMemSetAccess(dptr, size, &_accessDescr, 1)) != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemMap success,but cuMemSetAccess failed!, err code: %d\n", status);
        cuMemUnmap(dptr, size);
      }
    }
    // always release the handle, but the memory is accessible util cuMemUnmap
    if((status = cuMemRelease(allocationHandle)) != CUDA_SUCCESS) {
      fprintf(stderr, "cuMemRelease failed, err code: %d\n", status);
    } 
  } else {
    fprintf(stderr, "cuMemCreate %lx failed!, err code: %d\n", size, status);
  }
  return status == CUDA_SUCCESS ? 0 : -1;
}

// Free the physical memory [ptr, ptr + size]
static void freePhysicalMemory(void* ptr, size_t size) {
  CUdeviceptr dptr = (CUdeviceptr)ptr;
  CUresult res = cuMemUnmap(dptr, size); 
  if(res != CUDA_SUCCESS) {
    const char* errorStr;
    cuGetErrorString(res, &errorStr);
    fprintf(stderr, "cuMemUnmap failed when deallocating ptr %p and size %lx with error %s\n", ptr, size, errorStr);
  } 
}

/*
** kvCacheRegion functions implementation
*/
kvCacheRegion::kvCacheRegion(uint64_t region_size, uint64_t block_size, uint64_t page_size, CUdeviceptr ptr) {
  this->region_size = region_size;
  this->block_size = block_size;
  this->page_size = page_size; 
  this->dptr = reinterpret_cast<char*>(ptr);  
  this->nextUnmapedAddr = reinterpret_cast<char*>(ptr); 

  this->total_pages = 0;
  this->used_pages = 0; 
  this->alignedSize = 0;
}

// Decontructor: release all physical pages of this region
kvCacheRegion::~kvCacheRegion() {
  freeAllPhyMemory(); 
  // Note that since the region is detroyed, 
  // no need to clear other counters. 
}

CUdeviceptr kvCacheRegion::getStartPtr(void) {
  return reinterpret_cast<CUdeviceptr>(this->dptr); 
} 

/*
  kvCacheRegion function: allocate cached blocks  
    if the return value > 0, then it is succesful. 
 */ 
int64_t kvCacheRegion::allocCacheBlocks(uint64_t blocks, uint64_t * used_pages, cudaStream_t stream) {
  uint64_t size = blocks * this->block_size;

  int64_t toallocPages = -1; 

  // Align the new offset to page_size
  uint64_t alignedSize = roundup(size, this->page_size); 

  // No need to allocate if the cache is already larger than what we need
  if(alignedSize < this->alignedSize) {
    return 0; 
  }

  this->total_pages = alignedSize/this->page_size;

  // Updating the offset as we are using more blocks here. 
  this->alignedSize = alignedSize;
  
  // Check how many pages should we allocated this time
  char * alignedAddr = this->dptr + alignedSize; 
  if( alignedAddr > this->nextUnmapedAddr) {

    // Check whether alignedAddr is actually aligned well
    assert((alignedAddr - this->nextUnmapedAddr)%this->page_size == 0);
    toallocPages = (alignedAddr - this->nextUnmapedAddr)/this->page_size; 

    assert(toallocPages >= 0);

    uint64_t allocSize = toallocPages * this->page_size;

    // Allocate physical pages, which will exit if can't allocate successfully
    if (toallocPages > 0 && allocatePhyPages(this->nextUnmapedAddr, allocSize) == 0) {
      
      // Touch newly-allocates pages in order to initiate physical page allocation
      // This is important to avoid the memory allocation overhead on the critical path. 
      for(int i = 0; i < toallocPages; i++) {
        int64_t h_data = 0;
        int64_t offset = this->page_size * i;
        // Using different APIs for asynchronous memory allocations. 
        if(stream == nullptr) { 
          cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(this->nextUnmapedAddr + offset), &h_data, sizeof(int64_t));
        } else {
          cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(this->nextUnmapedAddr + offset), &h_data, sizeof(int64_t), stream);
        }
      }

      this->nextUnmapedAddr = alignedAddr;
      //fprintf(stderr, "%p : blocks %ld blocksize %lx this->alignedSize %lx this->nextUnmapedAddr %p alignedAddr %p\n", this->dptr, blocks, this->block_size, this->alignedSize, this->nextUnmapedAddr, alignedAddr);
      // Update the used pages correspondingly. The statement works even when this->offset is not aligned to page_size
      *used_pages += toallocPages; 
    }
  }
 
  return toallocPages; 
}

void kvCacheRegion::freeAllPhyMemory(void) {
  //fprintf(stderr, "freeAllPhyMemory dtpr %p alignedSize %lx\n", this->dptr, this->alignedSize);
  assert (this->alignedSize > 0);
  freePhysicalMemory(this->dptr, this->alignedSize);
  this->nextUnmapedAddr = this->dptr; 
  this->alignedSize = 0; 
}

/*
** kvCacheAllocator functions implementation
*/
kvCacheAllocator::kvCacheAllocator(int64_t max_seq_length, int64_t layers_num, int64_t heads_num, int64_t head_size, int64_t tokens_per_block, int64_t dtype_size) {
  uint64_t key_cache_block_per_layer =  tokens_per_block * heads_num * head_size * dtype_size; 
  uint64_t value_cache_block_per_layer = key_cache_block_per_layer;
  uint64_t cache_block_size = (key_cache_block_per_layer + value_cache_block_per_layer) * layers_num; 

  fprintf(stderr, "kvCacheAllocator initialization: key_cache_block_per_layer-%d, cache_block_size-%lx\n", key_cache_block_per_layer, cache_block_size); 
  // Getting the cuda device and force the initialization
  CUdevice dev; // device
  CHECK_RT(cudaFree(0));  // Force and check the initialization of the runtime
  CHECK_DRV(cuCtxGetDevice(&dev));
  
  size_t aligned_sz; 
  //_prop.type = CU_MEM_ALLOCATION_TYPE_MAX;
  _prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  //_prop.type = CU_MEM_ALLOCATION_TYPE_PORTABLE;
  _prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  _prop.location.id = dev;
  _accessDescr.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  _accessDescr.location = _prop.location;

  CHECK_DRV(cuMemGetAllocationGranularity(&aligned_sz, &_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  
  uint64_t max_blocks = roundup(max_seq_length, tokens_per_block)/tokens_per_block; 
  uint64_t region_size = max_blocks * cache_block_size * 2; 

  this->page_size = aligned_sz;
  this->region_size = ((region_size + aligned_sz - 1) / aligned_sz) * aligned_sz;
  this->block_size = cache_block_size;

  //printf("kvCacheAllocator: page_size-%ld, region_size-%ld, block_size-%ld\n", this->page_size, this->region_size, this->block_size);

  // TODO: finding out how much physical blocks it includes. This is just for the reference or watermark, as 
  // there is no need to rely on pre-assigned values if physical blocks are allocated on-demand
  size_t freeMem, totalMem;
  CHECK_RT(cudaMemGetInfo(&freeMem, &totalMem)); 

  this->watermark_pages = (((uint64_t)(freeMem * KV_UTILIZATION_RATE))/this->page_size);  
   
  // Doing other initialization
  this->total_pages = 0;
  this->used_pages = 0;
  this->active_regions = 0;

  this->manager_running = false;
  cuCtxGetCurrent(&origContext);

  cudaStreamCreate(&stream);

  // Initialize of mutex lock and condition
  pthread_mutex_init(&mutex_manager, NULL); 
  pthread_cond_init(&cond_manager, NULL); 
  manager_running = false; 

  pthread_attr_t attr; 
  pthread_attr_init(&attr);
  // Set the thread to be detached
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

  int result = pthread_create(&this->thread_id, &attr, kvCacheAllocator::memoryManagerThread, this);
  if(result != 0) {
    fprintf(stderr, "thread creation failed!"); 
    exit(0); 
  }
}

int64_t kvCacheAllocator::getPageSize() {
  return this->page_size;
}


// reserve function, reserve virtual address space for a request
int64_t kvCacheAllocator::reserveRegion(int64_t region_id) {
  CUdeviceptr ptr;
  kvCacheRegion * region = nullptr;

  // The expensive way to get a new region. Only invoked when no cached regions
  // Allocate the virtual address for this region
  CHECK_DRV(cuMemAddressReserve(&ptr, this->region_size, 0ULL, 0ULL, 0ULL));

  // Create a new region from the scratch
  region = new kvCacheRegion(this->region_size, this->block_size, this->page_size, ptr);

  // Allocate one block the first region
  if(region_id == 0) {
    uint64_t total_pages; 
    region->allocCacheBlocks(1, &total_pages, nullptr); 
  }

  std::lock_guard<std::mutex> lock(this->mutex);
  
  // Record the region information
  this->active_regions += 1; 
  this->active_regions_map[region_id] = region; 

  return static_cast<int64_t>(ptr);
}

// Release the region with the given region_id
void kvCacheAllocator::_releaseRegion(int64_t region_id) {
  // Find the region corresponding to the given region_id
  if(this->active_regions_map.count(region_id) == 0) {
    fprintf(stderr, "ERROR in release: region_id-%ld does not exist at all.!\n", region_id);
    exit(-1); 
  }

  std::lock_guard<std::mutex> lock(this->mutex);

  kvCacheRegion * region = this->active_regions_map[region_id];

  // Note that as we don't actually release physical cache blocks. 
  // Therefore, we don't need to change the active_blocks here. 
  region->freeAllPhyMemory();
  fprintf(stderr, "release region %ld\n", region_id); 
}

// alloc function, allocate physical memory, map to the reserved virtual address
// This function is designed for both prefill and decoding phase, where prefill may 
// require to save KV cache of multiple tokens, which should not invoke this function multiple times. 
// Similarly, the python code may get the physical blocks for multiple tokens during the decoding phase
// Note that the allocator doesn't care about tokens (which should be handled by the python code), but only blocks here.
int64_t kvCacheAllocator::_allocCacheBlocksForRequest(int64_t region_id, int64_t blocks, cudaStream_t stream) {
  int64_t pages = -1;

  CUresult result = cuCtxSetCurrent(origContext);
  if (result != CUDA_SUCCESS) {
      std::cerr << "Failed to set CUDA context in new thread: " << result << std::endl;
      return -1;
  }

  // Find the region corresponding to the given region_id, which should reserveRegion before
  // If the region_id doesn't exist at all, it is the bug that should be fixed.  
  if(this->active_regions_map.count(region_id) == 0) {
    fprintf(stderr, "ERROR in allocation: region_id %ld does not exist at all!\n", region_id);
    exit(-1); 
  }

  std::lock_guard<std::mutex> lock(this->mutex);

  kvCacheRegion * region = this->active_regions_map[region_id]; 

  pages = region->allocCacheBlocks(blocks, &this->used_pages, stream);

  return pages;
}

// Allocate cache blocks for a range of requests. Each request information will be an vector, with
// the request id as the first, and then number of blocks as the second. 
int64_t kvCacheAllocator::allocCacheBlocks(std::vector<std::vector<int64_t>> req_cache_blocks, cudaStream_t stream) {
  int64_t pages = 0; 

  for(auto row : req_cache_blocks) {
    uint64_t region_id = row[0]; 
    uint64_t blocks = row[1]; 

    pages += _allocCacheBlocksForRequest(region_id, blocks, stream);
    //if (region_id == 7)
    //fprintf(stderr, "allocate cache blocks for region-%d blocks %ld DONE\n", region_id, blocks);
  }
  //cudaDeviceSynchronize(); 

  return pages; 
}


void * kvCacheAllocator::memoryManagerThread(void * arg) {
  kvCacheAllocator * instance = static_cast<kvCacheAllocator *>(arg); 

  while(true) {
    pthread_mutex_lock(&instance->mutex_manager); 

    // We will wait if manager_running is true (didn't finish last memory management operations)
    // or there is no need to perform memory management
    while(!instance->manager_running) {
      pthread_cond_wait(&instance->cond_manager, &instance->mutex_manager); 
    }
  
    // Perform memory management asynchronously
    instance->swapOutCache(instance->swap_out_caches, instance->stream);
    instance->swapInCache(instance->swap_in_caches, instance->stream);
    instance->releaseRegions(instance->free_caches);
    instance->allocCacheBlocks(instance->req_cache_blocks, instance->stream);

    //pthread_mutex_lock(&instance->mutex_manager); 
    instance->manager_running = false; 
    pthread_cond_signal(&instance->cond_manager);
    pthread_mutex_unlock(&instance->mutex_manager); 
  }

  return NULL;
}
/* 
   This function mainly sets the work to be done, and then notify the manager thread to 
   perform memory management asynchronously. 
 */
void kvCacheAllocator::doAsyncKVCacheManage(std::vector<int64_t> free_caches, std::vector<std::vector<int64_t>> req_cache_blocks, 
        std::vector<std::vector<int64_t>> to_swap_out, std::vector<std::vector<int64_t>> to_swap_in) {
    pthread_mutex_lock(&this->mutex_manager);
    
    // If the manager has not finished, waiting on the condition 
    while(this->manager_running) {
      //fprintf(stderr, "waiting for the virtual memory management in asyn mode\n"); 
      pthread_cond_wait(&this->cond_manager, &this->mutex_manager); 
    }

    this->free_caches.clear(); 
    this->req_cache_blocks.clear();
    this->swap_out_caches.clear();  
    this->swap_in_caches.clear();  

    // Copying the work to the shared area
    for(auto cache_id: free_caches) {
      //fprintf(stderr, "releasing cache_id %d\n", cache_id); 
      this->free_caches.push_back(cache_id); 
    }

    for(auto cache_block: req_cache_blocks) {
      this->req_cache_blocks.push_back(cache_block); 
    }

    for(auto cacheInfo: to_swap_out) {
      this->swap_out_caches.push_back(cacheInfo); 
    }

    for(auto cacheInfo: to_swap_in) {
      this->swap_in_caches.push_back(cacheInfo); 
    }    
    
    this->manager_running = true; 
    pthread_cond_signal(&this->cond_manager); 
    pthread_mutex_unlock(&this->mutex_manager);
}

void kvCacheAllocator::updateCacheBlocks(bool immediate_allocate, std::vector<int64_t> free_caches, 
                                         std::vector<std::vector<int64_t>> req_cache_blocks,
                                         std::vector<std::vector<int64_t>> to_swap_out,
                                         std::vector<std::vector<int64_t>> to_swap_in) {
  //Py_BEGIN_ALLOW_THREADS
  //fprintf(stderr, "NNNNNNN immediate_allocate is %d\n", immediate_allocate); 

  if(immediate_allocate) {
    pthread_mutex_lock(&this->mutex_manager);
    
    // If the manager has not finished, waiting on the condition 
    while(this->manager_running) {
      pthread_cond_wait(&this->cond_manager, &this->mutex_manager); 
    }
    this->swapOutCache(to_swap_out, nullptr);
    this->swapInCache(to_swap_in, nullptr);
    this->releaseRegions(free_caches);
    this->allocCacheBlocks(req_cache_blocks, nullptr);

    pthread_mutex_unlock(&this->mutex_manager); 
  }
  else {
    doAsyncKVCacheManage(free_caches, req_cache_blocks, to_swap_out, to_swap_in);
  }
  //Py_END_ALLOW_THREADS
}

// Release regions specified in the vector
void kvCacheAllocator::releaseRegions(std::vector<int64_t> regions) {
  for(auto region : regions) {
    //fprintf(stderr, "release region-%d\n", region); 
    _releaseRegion(region);
  }
}

// Swap out the caches listed in src_to_dests (from Device to Host)
void kvCacheAllocator::swapOutCache(std::vector<std::vector<int64_t>> swap_caches, cudaStream_t stream) {
  
  for(auto item: swap_caches) {
    int64_t region_id = item[0]; 
    int64_t dest_ptr = item[1]; 
    int64_t size = item[2]; 

    kvCacheRegion * region = this->active_regions_map[region_id];
    CUdeviceptr src_ptr = region->getStartPtr(); 

    if(stream == nullptr) {
      cuMemcpyDtoH(reinterpret_cast<void*>(dest_ptr), src_ptr, size); 
    } else {
      cuMemcpyDtoHAsync(reinterpret_cast<void*>(dest_ptr), src_ptr, size, stream); 
    }
    
  }
}

// Swap in the caches listed in swap_caches (from Host to Device)
void kvCacheAllocator::swapInCache(std::vector<std::vector<int64_t>> swap_caches, cudaStream_t stream) {
    
  for(auto item: swap_caches) {
    int64_t src_ptr = item[0]; 
    int64_t region_id = item[1]; 
    int64_t blocks = item[2]; 

    // Allocate physical memory at first
    kvCacheRegion * region = this->active_regions_map[region_id];

    int64_t size = blocks  * this->block_size; 

    region->allocCacheBlocks(blocks+1, &this->used_pages, stream);
    
    CUdeviceptr dest_ptr = region->getStartPtr(); 
    fprintf(stderr, "SWPAIN src_ptr %lx, regionid-%ld, blocks %ld, address: %p, size: %lx\n", src_ptr, region_id, blocks, dest_ptr, size);

    if(stream == nullptr) {
      cuMemcpyHtoD(dest_ptr, reinterpret_cast<const void*>(src_ptr), size);
    }
    else {
      cuMemcpyHtoDAsync(dest_ptr, reinterpret_cast<const void*>(src_ptr), size, stream);
    }
  }

}

