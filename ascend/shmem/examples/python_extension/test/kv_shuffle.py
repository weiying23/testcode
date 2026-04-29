#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import multiprocessing
from dataclasses import dataclass
import torch
import torch_npu
import torch_npu.profiler


RANKS = 8

# Model Params
MAX_SEQLEN = 1024
MAX_BATCH = 10
INIT_BATCH = 5

# KVCache Params
PAGE_SIZE = 128
max_block_nums = MAX_SEQLEN * MAX_BATCH // PAGE_SIZE
KV_HEAD_NUM = 8
HEAD_DIM = 128


def get_pair_rank(sort_idx, local_rank):
    pair_rank = 0
    for idx, rank in enumerate(sort_idx):
        if rank == local_rank:
            pair_idx = (RANKS - 1) - idx # notice idx can't be RANKS
            pair_rank = sort_idx[pair_idx]
    return pair_rank.item()


max_transfer_tokens = 16384 * 2


def get_pair_transfer_tokens(kv_lens, kv_sum, kv_mean, pair_list, local_rank):
    pair_rank = pair_list[local_rank][0]
    if (kv_sum[local_rank] - kv_mean) <= 0:
        return -1, []
    if (kv_sum[local_rank] - kv_mean) >= 0 and (kv_sum[pair_rank] - kv_mean) >= 0:
        return -1, []
    transfer_tokens = min(abs(kv_sum[local_rank] - kv_mean), abs(kv_sum[pair_rank] - kv_mean))
    transfer_tokens = min(transfer_tokens, max_transfer_tokens)
    transfer_batch_id = []
    transfer_batch_tokens = 0
    for i in range(kv_lens[local_rank].shape[0]): # rank_loop
        if transfer_tokens > kv_lens[local_rank][i]: # 选择哪几个batch的block要搬到pair_rank
            transfer_batch_id.append(i)
            transfer_tokens -= kv_lens[local_rank][i]
            transfer_batch_tokens += kv_lens[local_rank][i]
    return transfer_batch_tokens, transfer_batch_id


def balance_kv(kv_lens):
    kv_sum = torch.sum(kv_lens, dim=-1) # (rank_size, 1)
    kv_mean = torch.mean(kv_sum.float(), dim=0).int() # int
    sort_kv_sum, sort_idx = torch.sort(kv_sum)

    # Get rank to rank pair
    pair_list = []
    for i in range(RANKS):
        pair_idx = get_pair_rank(sort_idx, i)
        pair_list.append([pair_idx])

    # Get rank to rank transfer_tokens
    transfer_tokens_list = []
    for i in range(RANKS):
        transfer_tokens, transfer_batch_id = get_pair_transfer_tokens(kv_lens, kv_sum, kv_mean, pair_list, i)
        if (transfer_tokens > 0):
            pair_list[i].append(0) # 0 means send
        else:
            pair_list[i].append(1) # 1 means recv
        transfer_tokens_list.append((transfer_tokens, transfer_batch_id))
    return pair_list, transfer_tokens_list


@dataclass
class CacheData:
    pair_list: list
    k_cache_list: list
    v_cache_list: list
    src_block_table: list
    dst_block_table: list
    block_num_list: list
    k_cache_list_g: list
    v_cache_list_g: list


def gendata(rank):
    torch.manual_seed(42)
    kv_lens = torch.randint(0, MAX_SEQLEN, (RANKS, INIT_BATCH))
    kv_sum = torch.sum(kv_lens, dim=-1) # (rank_size, 1)

    k_cache_list = []
    v_cache_list = []
    used_blocks_list = []
    batch_blocks_list = []
    # Prepare Inputs
    for i in range(RANKS):
        k_cache = torch.zeros((max_block_nums, KV_HEAD_NUM, PAGE_SIZE, HEAD_DIM), dtype=torch.int8)
        v_cache = torch.zeros((max_block_nums, KV_HEAD_NUM, PAGE_SIZE, HEAD_DIM), dtype=torch.int8)
        batch_list = kv_lens[i]
        used_id = 0
        batch_blocks = []
        for j in range(batch_list.shape[0]):
            seqlen = batch_list[j].item()
            block_num = seqlen // PAGE_SIZE + 1
            k_cache_real = torch.randint(low=-128, high=128, size=(block_num, KV_HEAD_NUM, PAGE_SIZE, HEAD_DIM),
                                         dtype=torch.int8)
            v_cache_real = torch.randint(low=-128, high=128, size=(block_num, KV_HEAD_NUM, PAGE_SIZE, HEAD_DIM),
                                         dtype=torch.int8)
            k_cache[used_id:used_id + block_num] = k_cache_real
            v_cache[used_id:used_id + block_num] = v_cache_real
            batch_blocks.append((j, [_ for _ in range(used_id, used_id + block_num)]))
            used_id += block_num
        used_blocks_list.append(used_id)
        batch_blocks_list.append(batch_blocks)
        k_cache_list.append(k_cache)
        v_cache_list.append(v_cache)

    # Shuffle Calculate
    pair_list, transfer_tokens_list = balance_kv(kv_lens)

    # Record Input KV Cache

    src_block_table = []
    dst_block_table = []
    block_num_list = []
    k_cache_list_g = [_k_cache.clone() for _k_cache in k_cache_list]
    v_cache_list_g = [_v_cache.clone() for _v_cache in v_cache_list]

    # Params Prepare And Golden Calculate
    for i in range(RANKS):
        local_rank = i
        shuffle_table = pair_list
        k_cache_list_g = k_cache_list_g
        v_cache_list_g = v_cache_list_g
        src_local_table = []
        dst_local_table = []

        transfer_batches = transfer_tokens_list[local_rank][1]
        for batch_idx in transfer_batches:
            src_local_table += batch_blocks_list[local_rank][batch_idx][1]

        pair_rank = shuffle_table[local_rank][0]
        for new_block_id in range(len(src_local_table)):
            dst_used_id = used_blocks_list[pair_rank]
            dst_local_table.append(dst_used_id + new_block_id)

        src_block_table.append(src_local_table)
        dst_block_table.append(dst_local_table)
        block_num_list.append(len(src_local_table))

        # Start KVCache Copy
        total_data_volume = 0
        if (shuffle_table[local_rank][1] == 0):
            for idx, _ in enumerate(src_local_table):
                src_idx = src_local_table[idx]
                dst_idx = dst_local_table[idx]
                k_cache_list_g[pair_rank][dst_idx] = k_cache_list_g[local_rank][src_idx]
                v_cache_list_g[pair_rank][dst_idx] = v_cache_list_g[local_rank][src_idx]

                def tensor_size_in_bytes(tensor):
                    num_elements = tensor.numel()
                    element_size = tensor.untyped_storage().element_size()
                    return num_elements * element_size

                data_volume = tensor_size_in_bytes(k_cache_list_g[local_rank][src_idx])
                total_data_volume += data_volume
                data_volume = tensor_size_in_bytes(v_cache_list_g[local_rank][src_idx])
                total_data_volume += data_volume
            if rank == 0:
                print(f"rank:{local_rank}, datasize{total_data_volume}!")

    return CacheData(pair_list, k_cache_list, v_cache_list, src_block_table,
                     dst_block_table, block_num_list, k_cache_list_g, v_cache_list_g)


def read_file(file_name):
    with open(file_name, 'rb') as file:
        file_content = file.read()

    # 获取文件内容的字节数
    num_bytes = len(file_content)
    return file_content, num_bytes


def worker(rank):
    # 加载共享库
    torch.ops.load_library('../output/lib/libshmem_torch.so')
    shmem_common = torch.classes.ShmemOps.Manager()
    torch_npu.npu.set_device(rank)
    stream = torch_npu.npu.Stream(device=f'npu:{torch_npu.npu.current_device()}')
    local_mem_size = 1024 * 1024 * 1024
    ipports = "tcp://127.0.0.1:8662"
    shmem_common.attr_init(rank, RANKS, local_mem_size, ipports)
    kv_shuffle = torch.classes.ShmemOps.KVShuffle()
    my_cache_data = gendata(rank)
    
    # global_shuffle_table
    global_shuffle_table = torch.tensor(my_cache_data.pair_list, dtype=torch.int64)
    global_shuffle_tensor = global_shuffle_table.npu()
    # k_cache
    k_cache = my_cache_data.k_cache_list[rank]
    shmem_k_cache_tensor = shmem_common.malloc_like(k_cache)
    # v_cache
    v_cache = my_cache_data.v_cache_list[rank]
    shmem_v_cache_tensor = shmem_common.malloc_like(v_cache)
    int64_data = my_cache_data.block_num_list[rank]
    
    # 检查第一个值是否为 0
    if int64_data != 0:
        # src_block_table
        src_block_table = torch.tensor(my_cache_data.src_block_table[rank], dtype=torch.int64)
        src_block_tensor = src_block_table.npu()

        # dst_block_table
        dst_block_table = torch.tensor(my_cache_data.dst_block_table[rank], dtype=torch.int64)
        dst_block_tensor = dst_block_table.npu()
    else:
        src_block_tensor = torch.Tensor()
        dst_block_tensor = torch.Tensor()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False
    )
    

    
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
            ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config) as prof:

        with torch_npu.npu.stream(stream):
            for _ in range(10):
                kv_shuffle.compute(global_shuffle_tensor, shmem_k_cache_tensor,
                                   shmem_v_cache_tensor, src_block_tensor, dst_block_tensor)
            stream.synchronize()
            prof.step()
    
    print("rank: ", rank, " kv_shuffle end !!!!")
    npu_tensork = shmem_k_cache_tensor.cpu()
    npu_tensorv = shmem_v_cache_tensor.cpu()
    print("K are equal:", torch.equal(npu_tensork, my_cache_data.k_cache_list_g[rank]))  # True
    print("V are equal:", torch.equal(npu_tensorv, my_cache_data.v_cache_list_g[rank]))  # True

    print("K are equal may be False:", torch.equal(npu_tensork, my_cache_data.k_cache_list[rank]))  # may be False
    print("V are equal may be False:", torch.equal(npu_tensorv, my_cache_data.v_cache_list[rank]))  # may be False

    shmem_common.free(shmem_k_cache_tensor)
    shmem_common.free(shmem_v_cache_tensor)


if __name__ == "__main__":
    num_processes = RANKS  # 定义要启动的进程数量
    processes = []

    for rank in range(num_processes):
        p = multiprocessing.Process(target=worker, args=(rank,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All processes have finished")