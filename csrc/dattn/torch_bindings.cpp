/*
 Copyright (c) ByteDance Inc.
 Authors: 
  - Tongping Liu (tongping.liu@bytedance.com)
  - https://github.com/vllm-project/vllm/pull/6102/commits
 */ 
#include "core/registration.h"
#include "dattn.h"


TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // kvCacheAllocator class bind
  m.class_<kvCacheAllocator>("kvCacheAllocator")
    .def(torch::init<int64_t, int64_t, int64_t>())
    .def("reserveRegion", &kvCacheAllocator::reserveRegion)
    .def("allocCPUCaches", &kvCacheAllocator::allocCPUCaches)
    .def("releaseRegions", &kvCacheAllocator::releaseRegions)
    .def("updateCacheBlocks", &kvCacheAllocator::updateCacheBlocks);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
