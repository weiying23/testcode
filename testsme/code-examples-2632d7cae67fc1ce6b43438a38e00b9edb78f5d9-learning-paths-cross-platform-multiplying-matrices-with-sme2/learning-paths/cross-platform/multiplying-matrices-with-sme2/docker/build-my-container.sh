#!/bin/bash
# SPDX-FileCopyrightText: Copyright 2024,2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: BSD-3-Clause-Clear

source assets.source_me

ARCH=$(uname -m)

# Fetch assets if we do not have them
if [ "${ARCH}" = "aarch64" ] || [ "${ARCH}" = "arm64" ]; then
  FVP_PKG=${FVP_PKG_ARM64}
  FVP_MODEL_DIR=${FVP_MODEL_DIR_ARM64}
  ATfE_PKG=${ATfE_PKG_ARM64}
  ATfE_DIR=${ATfE_DIR_ARM64}
elif [ "${ARCH}" = "x86_64" ]; then
  FVP_PKG=${FVP_PKG_X86}
  FVP_MODEL_DIR=${FVP_MODEL_DIR_X86}
  ATfE_PKG=${ATfE_PKG_X86}
  ATfE_DIR=${ATfE_DIR_X86}
else
  echo "Can not build docker image: '${ARCH}' is an unsupported host architecture"
  exit 1
fi

[ -d assets ] || mkdir assets
[ -f assets/${FVP_PKG} ] || wget -P assets/ ${FVP_BASE_URL}/${FVP_PKG}
[ -f assets/${ATfE_PKG} ] || wget -P assets/ ${ATfE_BASE_URL}/${ATfE_PKG}

# Build the container
docker build \
    --build-arg FVP_PKG=${FVP_PKG} \
    --build-arg FVP_MODEL_DIR=${FVP_MODEL_DIR} \
    --build-arg ATfE_PKG=${ATfE_PKG} \
    --build-arg ATfE_DIR=${ATfE_DIR} \
    --tag sme2-environment \
    --file sme2-environment.docker \
    assets
