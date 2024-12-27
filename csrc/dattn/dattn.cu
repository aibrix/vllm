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
#include <ATen/cuda/CUDAContext.h>
#include <Python.h>
#include <pthread.h>

#include "dattn.h"


#define KV_UTILIZATION_RATE (0.9)

constexpr int64_t MEGABYTES=1048576;
constexpr int64_t PAGE_SIZE=(MEGABYTES * 2); 
constexpr int64_t GIGABYTES=(MEGABYTES * 1024);

/* 
  In this allocator, we only have the following concepts, but without the concept of tokens.
  The python portion should convert the number of tokens to tokens depending on their cache_block_size (e.g., 16)
  Region: virtual address space for a request. Currently, we support the space for max_seq_len.
 */
static uint64_t roundup(uint64_t size, uint64_t align_size) {
  return ((size + align_size - 1)/align_size) * align_size; 
}

using PhysicalBlock = struct {
    CUmemGenericAllocationHandle handle;
};

class PhysicalBlocksManager {
public:
  // Available blocks will be placed in this pool 
  std::vector<PhysicalBlock> block_pool; 
  // All in-use blocks will be placed in the map. 
  std::unordered_map<void *, PhysicalBlock> block_map;
  int64_t block_size;
  int64_t free_blocks; 
  int64_t total_size;
  int64_t max_allowed_size; // maximum allowed size for KV cache
  int64_t page_size;  
  int64_t incremental_size; 
  int64_t tofree_blocks_watermark; 
  int64_t num_tofree_blocks; 
  CUmemAllocationProp prop;

  PhysicalBlocksManager(); 
  ~PhysicalBlocksManager();
  void initialize(size_t max_allowed_size, size_t total_memory, size_t block_size);

  PhysicalBlock allocate();
  void record(void * virtual_address, PhysicalBlock block);

  void free(void * virtual_address);

  void cleanup();

private:
  void _free_blocks_from_pool(int64_t num_blocks);
  void _increase_blocks(int64_t num_blocks);

}; 



static PhysicalBlocksManager _block_manager;

/*
 * In this file, there are three concepts:
    page_size: the actual page size of the underlying hardware, typically 2MB for cuda GPU
    cache_block_size: the original size of each block (16 tokens). However, this size (e.g., 5M) may not be aligned well with pages. 
    physical_block_size: the actual size used in managing of pysical blocks, which is consisted of multiple of cache_block_size. 
                      In particular, this size is algined with page_size.  
 */
PhysicalBlocksManager::PhysicalBlocksManager() {
  this->prop = {};
  this->prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  this->prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  this->prop.location.id = 0;

  // Each time, the size of physical blocks is to be increased whenever no objects is available
  this->incremental_size = 2 * GIGABYTES;
  this->total_size = 0;
  this->block_size = 0; 
  this->free_blocks = 0;  
}

PhysicalBlocksManager::~PhysicalBlocksManager() {
}

void PhysicalBlocksManager::_increase_blocks(int64_t num_blocks) {
    CUresult result;
    //fprintf(stderr, "_increase_blocks num_blocks-%d\n", num_blocks); 
    for (size_t i = 0; i < num_blocks; i++) {
        CUmemGenericAllocationHandle handle;
        result = cuMemCreate(&handle, this->block_size, &this->prop, 0);
        if (result != CUDA_SUCCESS) {
          fprintf(stderr, "Failed to create memory allocation, i-%d, with result %ld\n", i, result);
          exit(-1);        
        }

        block_pool.emplace_back(PhysicalBlock{handle});
    }
    // Update the number of blocks
    this->free_blocks += num_blocks; 
}

void PhysicalBlocksManager::initialize(size_t max_allowed_size, size_t total_memory, size_t block_size) {
    CUresult result;
    size_t page_size; 

    // Getting the granularity of page isze. 
    result = cuMemGetAllocationGranularity(&page_size, &this->prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to get page size");
    }

    assert(page_size == PAGE_SIZE); 
    assert(total_memory % GIGABYTES == 0);

    // We assume that cache_block_size is multiple of megabytes here.  

    // Allocate the initial blocks based on user's specification
    this->block_size = block_size; 

    this->max_allowed_size = max_allowed_size; 
    this->tofree_blocks_watermark = (this->incremental_size * 2)/block_size; 
    this->num_tofree_blocks = this->incremental_size/block_size;  

    // Allocate the physical memory with specified size 
    int64_t to_allocate_memory = min(total_memory, max_allowed_size); 
    this->total_size = to_allocate_memory; 
    size_t num_blocks = to_allocate_memory / block_size;
    fprintf(stderr, "total_memory %lx, max_allowed_size %lx num_blocks-%ld\n", total_memory, max_allowed_size, num_blocks);
    _increase_blocks(num_blocks);
}

PhysicalBlock PhysicalBlocksManager::allocate(void) {
    if(this->free_blocks == 0) {
      assert(this->block_pool.size() == 0); 

      // Keeping increase the memory if the GPU memory is sufficient
      int64_t allow_size = this->max_allowed_size - this->total_size;
      int64_t alloc_size; 
      //fprintf(stderr, "alloc_size %lx allow_size %lx total_size %lx\n", alloc_size, allow_size, this->total_size);

      if(allow_size <= 0) {
        fprintf(stderr, "There is no sufficent GPU memory now. ");
        exit(0);         
      }

      if (allow_size > this->incremental_size) {
        alloc_size = this->incremental_size; 
      }
      else {
        // Less than the incremental_size. 
        alloc_size = roundup(allow_size, this->block_size);  
      }

      int64_t blocks = alloc_size/this->block_size; 

      _increase_blocks(blocks);
      this->total_size += alloc_size; 
    }

    assert(this->free_blocks > 0); 
    assert(this->block_pool.size() > 0); 

    PhysicalBlock block = block_pool.back(); 
    block_pool.pop_back();     
    this->free_blocks--; 
    return block; 
}

void PhysicalBlocksManager::free(void * virtual_address) {
  PhysicalBlock block;
  bool is_exist = false; 

  if(block_map.count(virtual_address)) {
    block = block_map[virtual_address];

    block_map.erase(virtual_address); 
    is_exist = true; 
  }
  
  if (!is_exist) {
    fprintf(stderr, "Wrong: virtual_address-%p does not exist\n", virtual_address);
    exit(-1); 
  }

  // Adding this block to the block_pool
  block_pool.push_back(block); 
  this->free_blocks += 1; 

  if(block_pool.size() > this->tofree_blocks_watermark) {
    _free_blocks_from_pool(this->num_tofree_blocks);
  }
}

void PhysicalBlocksManager::_free_blocks_from_pool(int64_t num_blocks) {
    for(int i = 0; i < num_blocks; i++) {
      PhysicalBlock block = block_pool.back(); 
      CUresult status = CUDA_SUCCESS;
      if((status = cuMemRelease(block.handle)) != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemRelease failed, err code: %d\n", status);
      } 
      block_pool.pop_back(); 
    }

    this->free_blocks -= num_blocks; 
}

void PhysicalBlocksManager::cleanup() {
    for (auto& block : block_pool) {
      CUresult status = CUDA_SUCCESS;
      if((status = cuMemRelease(block.handle)) != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemRelease failed, err code: %d\n", status);
      } 
    }
    block_pool.clear();
}

void PhysicalBlocksManager::record(void * virtual_address, PhysicalBlock block) {
  block_map[virtual_address] = block; 
}

static CUmemAccessDesc _accessDescr = {};
 

/*
** kvCacheRegion functions implementation
*/
kvCacheRegion::kvCacheRegion(int64_t cache_block_size, int64_t physical_block_size, CUdeviceptr ptr) {
  this->cache_block_size = cache_block_size;
  this->physical_block_size = physical_block_size;
  this->dptr = reinterpret_cast<char*>(ptr);  
  this->nextUnmappedAddr = reinterpret_cast<char*>(ptr); 
  this->mapped_size = 0;
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
void kvCacheRegion::updateBlocks(uint64_t blocks, cudaStream_t stream) {
  uint64_t newSize = blocks * this->cache_block_size;
  newSize = roundup(newSize, this->physical_block_size); 

  int64_t distance; 

  // No need to allocate if size is not changed
  if(newSize == this->mapped_size) {
    return; 
  }
  else if (newSize < this->mapped_size) {
    // Shrink the memory for this region
    distance = this->mapped_size - newSize; 
    int64_t blocks_num = distance/this->physical_block_size; 

    char * addr = this->dptr + newSize; 
    this->nextUnmappedAddr = addr; 

    // Unmap unnecessary memory
    CUresult res; 
    res = cuMemUnmap(reinterpret_cast<CUdeviceptr>(addr), distance); 
    if(res != CUDA_SUCCESS) {
      const char* errorStr;
      cuGetErrorString(res, &errorStr);
      fprintf(stderr, "cuMemUnmap failed when deallocating ptr %p and size %lx with error %s\n", addr, distance, errorStr);
      exit(-1);
    }       

    //fprintf(stderr, "reduceBlocks, newSize: %lx, addr: %p, distance-%lx, blocks %ld, this->mapped_size: %lx \n", newSize, addr, distance, blocks_num, this->mapped_size);
    for(int i = 0; i < blocks_num; i++) {
      // Free the actual physical memory by putting it back to the pool
      _block_manager.free(addr); 

      addr += this->physical_block_size; 
    }
  }
  else {
    // Increase the memory for this region
    distance = newSize - this->mapped_size; 
    int64_t blocks_num = distance/this->physical_block_size; 

    char * addr = this->nextUnmappedAddr;

    //cudaDeviceSynchronize();

    // Map new memory
    CUresult res; 
    int64_t size = this->physical_block_size;  
    //fprintf(stderr, "increaseBlocks newSize: %lx, addr: %p, distance-%lx, blocks %ld, this->mapped_size: %lx\n", newSize, addr, distance, blocks_num, this->mapped_size);
    for(int i = 0; i < blocks_num; i++) {
      // Allocate a physical block 
      PhysicalBlock block = _block_manager.allocate();
      if ((res = cuMemMap(reinterpret_cast<CUdeviceptr>(addr), size, 0ULL, block.handle, 0ULL)) == CUDA_SUCCESS) {
        if ((res = cuMemSetAccess(reinterpret_cast<CUdeviceptr>(addr), size, &_accessDescr, 1)) != CUDA_SUCCESS) {
          fprintf(stderr, "cuMemMap success,but cuMemSetAccess failed!, err code: %d\n", res);
          cuMemUnmap(reinterpret_cast<CUdeviceptr>(addr), size);
          exit(-1);
        }
      }
      else {
        const char* errorStr;
        cuGetErrorString(res, &errorStr);
        fprintf(stderr, "cuMemMap failed when deallocating ptr %p res %d with error %s\n", addr, res, errorStr);
      }

      _block_manager.record(addr, block); 

      // Update addr to the next block
      addr += this->physical_block_size; 
    }
    this->nextUnmappedAddr = addr; 
  }

  this->mapped_size = newSize; 
}

void kvCacheRegion::freeAllPhyMemory(void) {
  //fprintf(stderr, "freeAllPhyMemory dtpr %p mapped_size %lx\n", this->dptr, this->mapped_size);
  assert (this->mapped_size > 0);

  char * addr = this->dptr;
  CUresult res = cuMemUnmap(reinterpret_cast<CUdeviceptr>(addr), this->mapped_size); 
  if(res != CUDA_SUCCESS) {
    const char* errorStr;
    cuGetErrorString(res, &errorStr);
    fprintf(stderr, "cuMemUnmap failed when deallocating ptr %p and size %lx with error %s\n", reinterpret_cast<CUdeviceptr>(addr), this->mapped_size, errorStr);
    exit(-1);
  }       

  int64_t blocks_num = this->mapped_size/this->physical_block_size; 
  for(int i = 0; i < blocks_num; i++) {
    // Free the actual physical memory by putting it back to the pool
    _block_manager.free(addr); 

    addr += this->physical_block_size; 
  }

  // Note that we don't actually release virtual memory (cuMemAddressFree)
  this->nextUnmappedAddr = this->dptr; 
  this->mapped_size = 0; 
}

/*
** kvCacheAllocator functions implementation
* TODO: we may need to remove some details from the allocator, such as max_seq_length, layers_num. 
*       But instead, we should add the initial allocation size, or we can use number of blocks (allocated size, so that )
*/
kvCacheAllocator::kvCacheAllocator(int64_t max_gpu_memory_size, int64_t cache_block_size, int64_t region_cache_size) {
  CUdevice device;
  //CHECK_DRV(cuInit(0));
  CHECK_RT(cudaFree(0));
  // Getting the cuda device and force the initialization
  CHECK_DRV(cuCtxGetDevice(&device));

  CHECK_DRV(cuCtxGetCurrent(&this->torchContext)); 
  //CHECK_DRV(cuCtxCreate(&this->origContext, 0, device));
  //CHECK_DRV(cuCtxSetCurrent(this->origContext));

  size_t free_memory, total_memory;
  CHECK_RT(cudaMemGetInfo(&free_memory, &total_memory)); 
  fprintf(stderr, "free_memory-%lx, total_memory-%lx\n", free_memory, total_memory);

  CUresult result;
  size_t page_size; 

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;

  _accessDescr.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  _accessDescr.location = prop.location;

  // Getting the granularity of page isze. 
  result = cuMemGetAllocationGranularity(&page_size, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (result != CUDA_SUCCESS) {
    fprintf(stderr, "Failed to get page size!\n");
    exit(-1);
  }

  fprintf(stderr, "kvCacheAllocator: max_gpu_memory_size - %lx, cache_block_size - %lx, region_cache_size - %lx, page_size - %lx\n", max_gpu_memory_size, cache_block_size, region_cache_size, page_size);
  assert(page_size == PAGE_SIZE); 

  int64_t to_allocate_memory = 2 * GIGABYTES; 
  if(free_memory < to_allocate_memory) {
    fprintf(stderr, "Insufficient gpu memory\n");
    exit(-1);
  }

  int64_t physical_block_size = cache_block_size; 

  while(physical_block_size%page_size != 0) {
    physical_block_size *= 2; 

    // Adding an explicit checking. 
    if(physical_block_size > 40*MEGABYTES) {
      fprintf(stderr, "Invalid physical_block_size %lx, with cache_block_size-%lx!!", physical_block_size, cache_block_size);
      exit(-1);
    }
  }
  this->physical_block_size = physical_block_size; 

  // Initialize block manager
  // max_allowed_size should be related to num_blocks, initialized GPU memory, cache_block_size
  _block_manager.initialize(max_gpu_memory_size, to_allocate_memory, physical_block_size);

  this->page_size = PAGE_SIZE;
  this->region_size = region_cache_size;
  this->cache_block_size = cache_block_size;

  this->manager_running = false;

  // Initialize of mutex lock and condition
  pthread_mutex_init(&mutex_manager, NULL); 
  pthread_cond_init(&cond_manager, NULL); 
  manager_running = false; 

  pthread_attr_t attr; 
  pthread_attr_init(&attr);
  // Set the thread to be detached
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

  if(pthread_create(&this->thread_id, &attr, kvCacheAllocator::memoryManagerThread, this) != 0) {
    fprintf(stderr, "thread creation failed!"); 
    exit(-1); 
  }

  CHECK_DRV(cuCtxSetCurrent(this->torchContext));
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
  region = new kvCacheRegion(this->cache_block_size, this->physical_block_size, ptr);

  // Allocate one block the first region
  if(region_id == 0) {
    region->updateBlocks(1, nullptr); 
  }

  // Record the region information
  this->active_regions_map[region_id] = region; 

  return static_cast<int64_t>(ptr);
}

std::vector<int64_t> kvCacheAllocator::allocCPUCaches(int64_t num_caches, int64_t cache_size) {
  std::vector<int64_t> cache_addresses; 
  
  for(int i = 0; i < num_caches; i++) {
    void * address;  
    cudaError_t err = cudaHostAlloc(&address, cache_size, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        std::cerr << "cudaHostAlloc failed: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
    cache_addresses.push_back((int64_t)address); 
  }

  return cache_addresses; 
}

// Release the region with the given region_id
void kvCacheAllocator::_releaseRegion(int64_t region_id) {
  // Find the region corresponding to the given region_id
  if(this->active_regions_map.count(region_id) == 0) {
    fprintf(stderr, "ERROR in release: region_id-%ld does not exist at all.!\n", region_id);
    exit(-1); 
  }

  //std::lock_guard<std::mutex> lock(this->mutex);
  kvCacheRegion * region = this->active_regions_map[region_id];

  //fprintf(stderr, "before release region %ld, blocks %d\n", region_id, _block_manager.block_pool.size()); 
  // Note that as we don't actually release physical cache blocks. 
  // Therefore, we don't need to change the active_blocks here. 
  region->freeAllPhyMemory();
  //fprintf(stderr, "after release region %ld, blocks %d\n", region_id, _block_manager.block_pool.size()); 
  //fprintf(stderr, "release region %ld\n", region_id); 
}


// Allocate cache blocks for a range of requests. Each request information will be an vector, with
// the request id as the first, and then number of blocks as the second. 
void kvCacheAllocator::updateBlocks(std::vector<std::vector<int64_t>> update_blocks, cudaStream_t stream) {
  for(auto row : update_blocks) {
    uint64_t region_id = row[0]; 
    uint64_t blocks = row[1]; 

    //fprintf(stderr, "region-%ld allocates %ld blocks. free_blocks-%d\n", region_id, blocks, _block_manager.block_pool.size()); 
    assert(this->active_regions_map.count(region_id) > 0);
    kvCacheRegion * region = this->active_regions_map[region_id];
    region->updateBlocks(blocks, stream);
    //fprintf(stderr, "after region-%ld allocates %ld blocks. free_blocks-%ld\n", region_id, blocks, _block_manager.block_pool.size()); 
    //fprintf(stderr, "region-%ld allocates %ld blocks. physical block size:%lx\n", region_id, blocks, this->physical_block_size); 
  }

  // Make sure that the asynchronized memcopy has finished. 
  if(stream)
    cudaStreamSynchronize(stream);

  //fprintf(stderr, "NNNNNN after updateBlocks, handling %ld request\n", update_blocks.size());
  return; 
}

// This is a separate thread that performing both synchronous and asynchronous 
// memory management operations. 
void * kvCacheAllocator::memoryManagerThread(void * arg) {
  kvCacheAllocator * instance = static_cast<kvCacheAllocator *>(arg); 
  
  // It is required to set current context if we are going 
  //CUresult result = cuCtxSetCurrent(instance->origContext);
  
  //cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaStream_t asyncStream;
  cudaStreamCreate(&asyncStream);

  while(true) {
    pthread_mutex_lock(&instance->mutex_manager); 

    // We will wait if manager_running is true (didn't finish last memory management operations)
    // or there is no need to perform memory management
    while(!instance->manager_running) {
      pthread_cond_wait(&instance->cond_manager, &instance->mutex_manager); 
    }

    cudaStream_t stream = nullptr; 
    // We will use a different stream for asynchronous operations. 
    if(!instance->immediate_allocate) {
      //stream = asyncStream;
      stream = at::cuda::getCurrentCUDAStream();
    } 

    //fprintf(stderr, "NNNNNNNNN in handling the request!!!!!\n");
    // Perform memory management asynchronously
    instance->swapOutCache(instance->swap_out_caches, stream);
    instance->updateBlocks(instance->update_blocks, stream);
    // Swap in cache must be done after allocating cache blocks, as 
    // we may reuse an existing cache but with the expansion of its blocks 
    instance->swapInCache(instance->swap_in_caches, stream);

    //pthread_mutex_lock(&instance->mutex_manager); 
    instance->manager_running = false; 
    pthread_cond_signal(&instance->cond_manager);
    pthread_mutex_unlock(&instance->mutex_manager); 
  }

  return NULL;
}

void kvCacheAllocator::updateCacheBlocks(bool immediate_allocate, 
                                         std::vector<std::vector<int64_t>> to_update_blocks,
                                         std::vector<std::vector<int64_t>> to_swap_out,
                                         std::vector<std::vector<int64_t>> to_swap_in) {
    //fprintf(stderr, "NNNNNNNNN in handling the request updateCacheBlocks!!!!!, immediate_allocate-%d\n", immediate_allocate);
    pthread_mutex_lock(&this->mutex_manager);
    
    // If the manager has not finished, waiting on the condition 
    while(this->manager_running) {
      //fprintf(stderr, "waiting for the virtual memory management in asyn mode\n"); 
      pthread_cond_wait(&this->cond_manager, &this->mutex_manager); 
    }

    this->update_blocks.clear();
    this->swap_out_caches.clear();  
    this->swap_in_caches.clear();  

    for(auto cache_block: to_update_blocks) {
      this->update_blocks.push_back(cache_block); 
    }

    for(auto cacheInfo: to_swap_out) {
      this->swap_out_caches.push_back(cacheInfo); 
    }

    for(auto cacheInfo: to_swap_in) {
      this->swap_in_caches.push_back(cacheInfo); 
    }    
    
    this->manager_running = true; 
    this->immediate_allocate = immediate_allocate; 
    pthread_cond_signal(&this->cond_manager); 
    pthread_mutex_unlock(&this->mutex_manager);

    if(immediate_allocate) {
      // We will wait until the manager thread finishes its job
      pthread_mutex_lock(&this->mutex_manager);
      while(this->manager_running) {
        //fprintf(stderr, "waiting for the virtual memory management in asyn mode\n"); 
        pthread_cond_wait(&this->cond_manager, &this->mutex_manager); 
      } 
      pthread_mutex_unlock(&this->mutex_manager); 
    }
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

    assert(this->active_regions_map.count(region_id) != 0);

    kvCacheRegion * region = this->active_regions_map[region_id];
    CUdeviceptr src_ptr = region->getStartPtr(); 
    
    //cuMemcpyDtoH(reinterpret_cast<void*>(dest_ptr), src_ptr, size); 
    //continue;

    if(stream == nullptr) {
      cuMemcpyDtoH(reinterpret_cast<void*>(dest_ptr), src_ptr, size); 
    } else {
      //cudaMemcpyAsync(reinterpret_cast<void*>(dest_ptr), reinterpret_cast<void*>(src_ptr), size, cudaMemcpyDeviceToHost, nullptr);
      //cudaMemcpyAsync(reinterpret_cast<void*>(dest_ptr), reinterpret_cast<void*>(src_ptr), size, cudaMemcpyDeviceToHost, stream);
      cuMemcpyDtoHAsync(reinterpret_cast<void*>(dest_ptr), src_ptr, size, stream); 

      //fprintf(stderr, "swaping out region-%ld, src-%p, dest-%p, size-%lx\n", region_id, src_ptr, dest_ptr, size);
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

    int64_t size = blocks  * this->cache_block_size; 

    //fprintf(stderr, "SWPAIN allocation regionid-%ld, blocks %ld, size: %lx\n", region_id, blocks, size);
    // NOTE: no need to updateBlocks, as we have done that before
    //region->updateBlocks(blocks, stream);
    
    CUdeviceptr dest_ptr = region->getStartPtr(); 

    if(stream == nullptr) {
      cuMemcpyHtoD(dest_ptr, reinterpret_cast<const void*>(src_ptr), size);
    }
    else {
      cuMemcpyHtoDAsync(dest_ptr, reinterpret_cast<const void*>(src_ptr), size, stream);
    }
  }
}