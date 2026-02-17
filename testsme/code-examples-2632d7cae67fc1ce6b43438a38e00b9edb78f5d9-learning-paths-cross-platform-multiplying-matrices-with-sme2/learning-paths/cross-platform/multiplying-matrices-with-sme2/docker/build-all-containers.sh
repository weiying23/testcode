#!/bin/bash
# SPDX-FileCopyrightText: Copyright 2024,2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: BSD-3-Clause-Clear

set -e

usage()
{
cat << EOF
Builds and optionally publishes the environment multi-arch container for the SME2 learning path
(x86_64 and aarch64 are currently supported).

Usage:
$(basename $0) --version <version> [--repository <repo>] [--publish]

Where:

  --version     mandatory option to set the image version. Something like "v1".
  --repository  the repository name to use (*including* the trailing "/" !)
  --publish     will make the script push the image to the registry.

  -h, --help    this help message
EOF
}

# Default option values.
REPOSITORY=
VERSION=
PUBLISH=0

# Parse command line.
while [ $# -gt 0 ]; do
        case $1 in
        --repository)
                REPOSITORY="$2"
                shift # past argument
                shift # past value
                ;;
        --version)
                VERSION="$2"
                shift # past argument
                shift # past value
                ;;
        --publish)
                PUBLISH=1
                shift # past argument
                ;;
        -h|--help)
                usage
                exit 0
                ;;
        -*|--*)
                echo "error: unexpected named argument! ($1)"
                usage
                exit 1
                ;;
        *)
                echo "error: unexpected positional argument! ($1)"
                usage
                exit 1
                ;;
        esac
done

# VERSION is mandatory.
echo "Version: $VERSION"
if [ -z "${VERSION}" ]; then
  echo "error: version is not set"
  usage
  exit 1
fi

source assets.source_me

# Fetch all assets that we do not yet have
[ -d assets ] || mkdir assets
[ -f assets/${FVP_PKG_ARM64} ] || wget -P assets/ ${FVP_BASE_URL}/${FVP_PKG_ARM64}
[ -f assets/${ATfE_PKG_ARM64} ] || wget -P assets/ ${ATfE_BASE_URL}/${ATfE_PKG_ARM64}
[ -f assets/${FVP_PKG_X86} ] || wget -P assets/ ${FVP_BASE_URL}/${FVP_PKG_X86}
[ -f assets/${ATfE_PKG_X86} ] || wget -P assets/ ${ATfE_BASE_URL}/${ATfE_PKG_X86}

# Build the container for aarch64 and x86
echo
echo "### Building ${REPOSITORY}sme2-learning-path:sme2-environment-${VERSION}-aarch64"
docker buildx build \
    --build-arg FVP_PKG=${FVP_PKG_ARM64} \
    --build-arg FVP_MODEL_DIR=${FVP_MODEL_DIR_ARM64} \
    --build-arg ATfE_PKG=${ATfE_PKG_ARM64} \
    --build-arg ATfE_DIR=${ATfE_DIR_ARM64} \
    --platform linux/arm64/v8 \
    --tag ${REPOSITORY}sme2-learning-path:sme2-environment-${VERSION}-aarch64 \
    --file sme2-environment.docker \
    assets

echo
echo "### ${REPOSITORY}sme2-learning-path:sme2-environment-${VERSION}-x86_64"
docker buildx build \
    --build-arg FVP_PKG=${FVP_PKG_X86} \
    --build-arg FVP_MODEL_DIR=${FVP_MODEL_DIR_X86} \
    --build-arg ATfE_PKG=${ATfE_PKG_X86} \
    --build-arg ATfE_DIR=${ATfE_DIR_X86} \
    --platform linux/amd64 \
    --tag ${REPOSITORY}sme2-learning-path:sme2-environment-${VERSION}-x86_64 \
    --file sme2-environment.docker \
    assets

if [ ${PUBLISH} -ne 0 ]; then
  echo
  echo "### Pushing image ${REPOSITORY}sme2-learning-path:sme2-environment-${VERSION}-aarch64"
  docker push ${REPOSITORY}sme2-learning-path:sme2-environment-${VERSION}-aarch64
  echo "### Pushing image ${REPOSITORY}sme2-learning-path:sme2-environment-${VERSION}-x86_64"
  docker push ${REPOSITORY}sme2-learning-path:sme2-environment-${VERSION}-x86_64

  # Build the manifest to get a multi-arch image.
  echo
  echo "### Building manifest ${REPOSITORY}sme2-learning-path:sme2-environment-${VERSION}"
  docker manifest create ${REPOSITORY}sme2-learning-path:sme2-environment-${VERSION} \
      --amend ${REPOSITORY}sme2-learning-path:sme2-environment-${VERSION}-x86_64 \
      --amend ${REPOSITORY}sme2-learning-path:sme2-environment-${VERSION}-aarch64

  echo
  echo "### Pushing multi-arch image ${REPOSITORY}sme2-learning-path:sme2-environment-${VERSION}"
  docker manifest push --purge ${REPOSITORY}sme2-learning-path:sme2-environment-${VERSION}
fi
