/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <vector>
#include <string>
#include "torch_npu/csrc/aten/common/from_blob.h"

#include "shmem_api.h"
#include "shmem_kernel.h"

namespace ShmemOps {

void print_tensor_info(const at::Tensor& tensor)
{
    // 打印张量的形状
    std::cout << "Shape: ";
    for (int64_t dim : tensor.sizes()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    // 打印张量的数据类型
    std::cout << "Dtype: " << tensor.dtype() << std::endl;

    // 打印张量的首地址
    if (tensor.defined()) {
        std::cout << "First element address: " << reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) << std::endl;
    } else {
        std::cout << "Tensor is not defined." << std::endl;
    }
}

class Manager : public torch::jit::CustomClassHolder {
public:
    // 默认构造函数
    Manager() : name_("Manager") {}

    std::string get_name() const
    {
        return name_;
    }

    int64_t attr_init(int64_t my_rank, int64_t n_ranks, int64_t local_mem_size, const std::string& ip_port)
    {
        int64_t status = 0;
        shmem_init_attr_t *attributes;
        status = shmem_set_conf_store_tls(false, nullptr, 0);
        status = shmem_set_attr(my_rank, n_ranks, local_mem_size, ip_port.c_str(), &attributes);
        status = shmem_init_attr(attributes);
        return status;
    }
    int64_t finalize()
    {
        return shmem_finalize();
    }
    
    at::Tensor malloc_tensor(int64_t size)
    {
        void *symmPtr = shmem_malloc(size);
        at::Tensor shmem_tensor = at_npu::native::from_blob(symmPtr, size, torch::dtype(torch::kUInt8));

        return shmem_tensor;
    }

    at::Tensor malloc_like(const at::Tensor& npu_tensor)
    {
        void *npu_tensor_ptr = static_cast<void *>(const_cast<void *>(npu_tensor.storage().data()));
        int64_t size = npu_tensor.storage().nbytes();
        void *symmPtr = shmem_malloc(size);
        at::Tensor shmem_tensor = at_npu::native::from_blob(symmPtr, npu_tensor.sizes(), npu_tensor.dtype());
        aclrtMemcpy(symmPtr, size, npu_tensor_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE);
        return shmem_tensor;
    }

    void free_tensor(const at::Tensor& shmem_tensor)
    {
        void *shmem_ptr = static_cast<void *>(const_cast<void *>(shmem_tensor.storage().data()));
        shmem_free(shmem_ptr);
        return;
    }

private:
    std::string name_;
};

static constexpr uint32_t DEFAULT_BLOCK_DIM = 16;
class KVShuffle : public torch::jit::CustomClassHolder {
public:
    // 默认构造函数
    KVShuffle() : name_("ShmemKVShuffle"), count_(0), block_dim_(DEFAULT_BLOCK_DIM)
    {
        fftsAddr_ = shmemx_get_ffts_config();
        int64_t SYNC_FLAG_INTERVAL = 16;
        sync_ptr_ = shmem_malloc(sizeof(int32_t) * shmem_n_pes() * block_dim_ * SYNC_FLAG_INTERVAL);
        aclrtMemset(sync_ptr_, sizeof(int32_t) * shmem_n_pes() * block_dim_  * SYNC_FLAG_INTERVAL, 0,
                    sizeof(int32_t) * shmem_n_pes() * block_dim_ * SYNC_FLAG_INTERVAL);
    }

    ~KVShuffle()
    {
        shmem_free(sync_ptr_);
    }

    std::string get_name() const
    {
        return name_;
    }
    
    void compute(const at::Tensor &ShuffleTbale, const at::Tensor &KeyCache, const at::Tensor &ValueCache,
                 const at::Tensor &SrcBlockTable, const at::Tensor &DstBlockTable)
    {
        void *global_shuffle_table = const_cast<void *>(ShuffleTbale.storage().data());

        void *k_cache = const_cast<void *>(KeyCache.storage().data());

        void *v_cache = const_cast<void *>(ValueCache.storage().data());

        void *src_block_table = const_cast<void *>(SrcBlockTable.storage().data());
        void *dst_block_table = const_cast<void *>(DstBlockTable.storage().data());

        int64_t block_nums = DstBlockTable.size(0);
        int64_t kv_head_num = KeyCache.size(1);
        int64_t head_dim = KeyCache.size(3);
        int64_t page_size = KeyCache.size(2);

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        count_++;
        ShmemKernel::shmem_kv_shuffle(block_dim_, stream, fftsAddr_, k_cache,
            v_cache, global_shuffle_table, src_block_table, dst_block_table, sync_ptr_,
            block_nums, kv_head_num, page_size, head_dim, count_);
    }
private:
    std::string name_;
    int32_t count_;
    uint32_t block_dim_;
    uint64_t fftsAddr_;
    void* sync_ptr_;
};
}  // namespace ShmemOps

// 注册类到 PyTorch JIT 系统
// 注册类到 PyTorch JIT 系统
static auto registry_common = torch::jit::class_<ShmemOps::Manager>("ShmemOps", "Manager")
    .def(torch::jit::init<>()) // 默认构造函数
    .def("attr_init", &ShmemOps::Manager::attr_init)
    .def("finalize", &ShmemOps::Manager::finalize)
    .def("malloc", &ShmemOps::Manager::malloc_tensor)
    .def("free", &ShmemOps::Manager::free_tensor)
    .def("malloc_like", &ShmemOps::Manager::malloc_like)
    .def("get_name", &ShmemOps::Manager::get_name);

static auto registry_kvshuffle = torch::jit::class_<ShmemOps::KVShuffle>("ShmemOps", "KVShuffle")
    .def(torch::jit::init<>()) // 默认构造函数数
    .def("compute", &ShmemOps::KVShuffle::compute)
    .def("get_name", &ShmemOps::KVShuffle::get_name);
