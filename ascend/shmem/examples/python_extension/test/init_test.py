#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
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
G_IP_PORT = "tcp://127.0.0.1:8666"


def run_tests():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # 1. test set tls info
    ret = ash.set_conf_store_tls(False, "")
    if ret != 0:
        raise ValueError("[ERROR] set_conf_store_tls failed")
    # 2. test init
    attributes = ash.InitAttr()
    attributes.my_rank = rank
    attributes.n_ranks = world_size
    attributes.local_mem_size = g_ash_size
    attributes.ip_port = G_IP_PORT
    attributes.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    ret = ash.shmem_init(attributes)
    if ret != 0:
        raise ValueError('[ERROR] shmem_init failed')
    # 3. test malloc
    shmem_ptr = ash.shmem_malloc(g_malloc_size)
    print(f'rank[{rank}]: shmem_ptr:{shmem_ptr} with type{type(shmem_ptr)}')
    if shmem_ptr is None:
        raise ValueError('[ERROR] shmem_malloc failed')
    _ = ash.shmem_free(shmem_ptr)
    # 4. test pe
    my_pe, pe_count = ash.my_pe(), ash.pe_count()
    print(f'rank[{rank}]: my_pe:{my_pe} and pe_count:{pe_count}')
    if not (my_pe == rank and pe_count == world_size):
        raise ValueError('[ERROR] pe/world failed')
    # 5. test team
    my_team_pe, team_pe_count = ash.team_my_pe(0), ash.team_n_pes(0)
    print(f'x: rank[{rank}]: t_my_pe:{my_team_pe} and t_pe_count:{team_pe_count}')

    # 6. test finialize
    _ = ash.shmem_finialize()

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    # shmem init must comes after torch.npu.set_device(or any other aclInit device action)
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", rank=local_rank)
    run_tests()
    print("init_test.py running success!")
