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

"""python api for shmem."""

import os

import setuptools
from setuptools import find_namespace_packages
from setuptools.dist import Distribution

# 消除whl压缩包的时间戳差异
os.environ['SOURCE_DATE_EPOCH'] = '0'

current_version = os.getenv('VERSION', '1.0.0')


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(self):
        return True


setuptools.setup(
    name="shmem",
    version=current_version,
    author="",
    author_email="",
    description="python api for shmem",
    packages=find_namespace_packages(exclude=("tests*",)),
    license="Apache License Version 2.0",
    install_requires=["torch-npu"],
    python_requires=">=3.7",
    package_data={"shmem": ["_pyshmem.cpython*.so", "lib/**", "VERSION"]},
    distclass=BinaryDistribution
)
