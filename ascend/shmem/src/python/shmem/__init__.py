#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import os
import sys
import ctypes
import logging
import torch_npu
from ._pyshmem import (shmem_init, shmem_get_unique_id, shmem_init_using_unique_id, \
                       shmem_finialize, shmem_malloc, shmem_free, \
                       shmem_ptr, my_pe, pe_count, set_conf_store_tls_key, mte_set_ub_params, team_split_strided,
                       team_split_2d, team_translate_pe, \
                       team_destroy, InitAttr, OpEngineType, shmem_set_attributes, shmem_set_data_op_engine_type,
                       shmem_set_timeout, \
                       InitStatus, shmem_calloc, shmem_align, shmem_init_status, get_ffts_config, team_my_pe,
                       team_n_pes, \
                       shmem_putmem_nbi, shmem_getmem_nbi, shmem_putmem, shmem_getmem, shmem_putmem_signal,
                       shmem_putmem_signal_nbi, \
                       shmem_info_get_version, shmem_info_get_name, \
                       shmem_team_get_config, OptionalAttr, shmem_global_exit, set_conf_store_tls, set_log_level,
                       set_extern_logger)

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
sys.path.append(current_dir)
libs_path = os.path.join(os.getenv('SHMEM_HOME_PATH', current_dir), 'shmem/lib')
for lib in ["libshmem.so"]:
    lib_path = os.path.join(libs_path, lib)
    try:
        ctypes.CDLL(lib_path)
    except OSError as e:
        logging.error(f"Failed to load shared library: {lib_path}, {e}")

__all__ = [
    'shmem_init',
    'shmem_get_unique_id',
    'shmem_init_using_unique_id',
    'shmem_finialize',
    'shmem_malloc',
    'shmem_free',
    'shmem_ptr',
    'my_pe',
    'pe_count',
    'set_conf_store_tls_key',
    'set_conf_store_tls',
    'team_my_pe',
    'team_n_pes',
    'mte_set_ub_params',
    'team_split_strided',
    'team_split_2d',
    'team_translate_pe',
    'team_destroy',
    'InitAttr',
    'InitStatus',
    'OpEngineType',
    'shmem_set_attributes',
    'shmem_set_data_op_engine_type',
    'shmem_set_timeout',
    'shmem_calloc',
    'shmem_align',
    'shmem_init_status',
    'get_ffts_config',
    'shmem_global_exit',
    'shmem_putmem_nbi',
    'shmem_getmem_nbi',
    'shmem_putmem_signal',
    'shmem_putmem_signal_nbi',
    'shmem_putmem',
    'shmem_getmem',
    'shmem_info_get_version',
    'shmem_info_get_name',
    'shmem_team_get_config',
    'set_log_level',
    'set_extern_logger'
]