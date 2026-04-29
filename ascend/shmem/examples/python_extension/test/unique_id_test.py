#
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import os
import torch
import torch.distributed as dist
import torch_npu
import shmem as ash


g_ash_size = 1024 * 1024 * 1024
g_malloc_size = 8 * 1024 * 1024


def run_init_with_unique_id_tests():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ret = ash.set_conf_store_tls(False, "")

    # 0. disabel TLS
    if ret != 0:
        raise ValueError("[ERROR] disable tls failed.")

    # 1. get unique id
    uid_size = 512
    tensor = torch.zeros(uid_size, dtype=torch.uint8)
    if rank == 0:
        unique_id = ash.shmem_get_unique_id()
        if unique_id is None:
            raise ValueError('[ERROR] get unique id failed')
        tensor = torch.tensor(list(unique_id), dtype=torch.uint8)
    dist.broadcast(tensor, src=0)
    if rank != 0:
        unique_id = bytes(tensor.tolist())
    # 2. init with unique id
    ret = ash.shmem_init_using_unique_id(rank, world_size, g_ash_size, unique_id)
    if ret != 0:
        raise ValueError('[ERROR] shmem_init failed')

    # test malloc
    shmem_ptr = ash.shmem_malloc(g_malloc_size)
    print(f'rank[{rank}]: shmem_ptr:{shmem_ptr} with type{type(shmem_ptr)}')
    if shmem_ptr is None:
        raise ValueError('[ERROR] shmem_malloc failed')

    # test pe
    my_pe, pe_count = ash.my_pe(), ash.pe_count()
    print(f'rank[{rank}]: my_pe:{my_pe} and pe_count:{pe_count}')
    if not (my_pe == rank and pe_count == world_size):
        raise ValueError('[ERROR] pe/world failed')

    # test free
    _ = ash.shmem_free(shmem_ptr)

    # test finialize
    _ = ash.shmem_finialize()


if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)

    dist.init_process_group(backend="gloo", init_method="env://")
    run_init_with_unique_id_tests()
    print("test.py running success!")