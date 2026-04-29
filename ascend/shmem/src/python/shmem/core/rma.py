#!/usr/bin/env python
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
import logging

import shmem._pyshmem as _pyshmem
from shmem.core.utils import Buffer, AclshmemInvalid
from shmem.core.direct import ComparisonType, SignalOp

__all__ = ['put_signal', 'signal_op', 'signal_wait', 'put', 'get']

logger = logging.getLogger("aclshmem")


def put_signal(dst: Buffer, src: Buffer, signal_var: Buffer, signal_val: int, signal_operation: SignalOp,
               remote_pe: int=-1, stream=None) -> None:
    """
    Perform a put-with-signal operation on an NPU stream.

    Args:
        dst (Buffer): Symmetric memory buffer on the local PE (destination).
        src (Buffer): Symmetric memory buffer containing the source data.
        signal_var (Buffer): Symmetric memory buffer used as the signal variable.
        signal_operation (SignalOp): Operation applied to update the signal variable.
        remote_pe (int): PE number of the remote processing element.
        stream: Reserved parameter. **Ignored** in ACLSHMEM and always treated as ``None``.
    """
    _pyshmem.aclshmemx_putmem_signal(
        dst.addr, src.addr, signal_var.length, signal_var.addr, signal_val, signal_operation, remote_pe
    )


def signal_op(signal_var: Buffer, signal_val: int, signal_operation: SignalOp, remote_pe: int=-1,
              stream: int=None) -> None:
    """
    Performs an atomic operation on a remote signal variable at the specified PE,
    with the operation executed on the given stream.

    Args:
        signal_var (Buffer): Symmetric memory buffer containing the signal variable,
                             accessible by the remote PE.
        signal_val (int): Value to be used in the remote atomic signal operation.
        signal_operation (SignalOp): Atomic operation to apply on the remote signal variable.
        remote_pe (int): PE number of the remote processing element.
        stream (int): ACL Stream object used for execution ordering.
                      Must be a valid stream created via ACL runtime.
    """
    if stream is None:
        logger.error("Signal operations without an explicit stream are not supported.")
        raise AclshmemInvalid("Signal operations without an explicit stream are not supported.")
    _pyshmem.aclshmemx_signal_op_on_stream(signal_var.addr, signal_val, signal_operation, remote_pe, stream)


def signal_wait(signal_var: Buffer, signal_val: int, signal_operation: ComparisonType, stream: int=None) -> None:
    """
    Wait until a symmetric signal variable meets a specified condition.

    Args:
        signal_var (Buffer): Symmetric memory buffer containing the signal variable,
                             which is accessible across PEs.
        signal_val (int): Value to compare the signal variable against.
        signal_operation (ComparisonType): Comparison operation used to evaluate the condition
                                    (e.g., EQUAL, NOT_EQUAL, LESS_THAN, etc.).
        stream (int): ACL Stream object used for execution ordering.
                      Must be a valid stream created via the ACL runtime.
    """
    if stream is None:
        logger.error("signal_wait operations without an explicit stream are not supported.")
        raise AclshmemInvalid("signal_wait operations without an explicit stream are not supported.")

    _pyshmem.aclshmemx_signal_wait_until_on_stream(signal_var.addr, signal_operation, signal_val, stream)


def put(dst: Buffer, src: Buffer, remote_pe: int=-1, stream: int=None) -> None:
    """
    Copy contiguous data from the local PE to a symmetric memory address on a remote PE.

    Args:
        dst (Buffer): Symmetric destination buffer on the remote PE.
        src (Buffer): Local source buffer containing the data to send.
        remote_pe (int): PE number of the target processing element.
        stream (int): ACL Stream used to order the copy operation.
                      Must be a valid stream created via the ACL runtime.
    """
    _pyshmem.aclshmemx_putmem_on_stream(dst.addr, src.addr, src.length, remote_pe, stream)

def get(dst: Buffer, src: Buffer, remote_pe: int=-1, stream: int=None) -> None:
    """
    Copy contiguous data from symmetric memory on a remote PE to a local buffer.

    Args:
        dst (Buffer): Local destination buffer where data will be written.
        src (Buffer): Symmetric source buffer on the remote PE.
        remote_pe (int): PE number of the source processing element.
        stream (int): ACL Stream used to order the copy operation.
                      Must be a valid stream created via the ACL runtime.
    """
    _pyshmem.aclshmemx_getmem_on_stream(dst.addr, src.addr, src.length, remote_pe, stream)

