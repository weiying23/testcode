#!/bin/bash
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Exit on error.
set -e

if [ "$1" = "--trace" ]; then
  FVP_MODEL_DIR=$(dirname $(which FVP_Base_RevC-2xAEMvA))
  TarmacTrace_PLUGIN=$FVP_MODEL_DIR/../../plugins/$(basename $FVP_MODEL_DIR)/TarmacTrace.so
  TARMAC_TRACE="--plugin $TarmacTrace_PLUGIN -C TRACE.TarmacTrace.trace-file=trace.tarmac"
  PROG=$2
  PROG_ARGS=${@:3}
else
  TARMAC_TRACE=""
  PROG=$1
  PROG_ARGS=${@:2}
fi

FVP_Base_RevC-2xAEMvA \
    -C bp.secure_memory=0 \
    -C bp.ve_sysregs.exit_on_shutdown=1 \
    -C bp.vis.disable_visualisation=1 \
    -C cache_state_modelled=0 \
    -C cluster0.NUM_CORES=1 \
    -C cluster0.check_memory_attributes=0 \
    -C cluster0.ecv_support_level=2 \
    -C cluster0.enhanced_pac2_level=3 \
    -C cluster0.has_16k_granule=1 \
    -C cluster0.has_amu=1 \
    -C cluster0.has_arm_v8-1=1 \
    -C cluster0.has_arm_v8-2=1 \
    -C cluster0.has_arm_v8-3=1 \
    -C cluster0.has_arm_v8-4=1 \
    -C cluster0.has_arm_v8-5=1 \
    -C cluster0.has_arm_v8-6=1  \
    -C cluster0.has_arm_v8-7=1 \
    -C cluster0.has_arm_v8-8=1 \
    -C cluster0.has_arm_v8-9=1 \
    -C cluster0.has_arm_v9-0=1 \
    -C cluster0.has_arm_v9-1=1 \
    -C cluster0.has_arm_v9-2=1 \
    -C cluster0.has_arm_v9-3=1 \
    -C cluster0.has_arm_v9-4=1 \
    -C cluster0.has_branch_target_exception=1 \
    -C cluster0.has_brbe=1 \
    -C cluster0.has_brbe_v1p1=1 \
    -C cluster0.has_const_pac=1 \
    -C cluster0.has_gcs=1 \
    -C cluster0.has_hpmn0=1 \
    -C cluster0.has_large_system_ext=1 \
    -C cluster0.has_large_va=1 \
    -C cluster0.has_permission_indirection_s1=1 \
    -C cluster0.has_permission_indirection_s2=1 \
    -C cluster0.has_permission_overlay_s1=1 \
    -C cluster0.has_permission_overlay_s2=1 \
    -C cluster0.has_rndr=1 \
    -C cluster0.has_sve=1 \
    -C cluster0.max_32bit_el=0 \
    -C cluster0.pmb_idr_external_abort=1 \
    -C cluster0.sve.has_sme2=1 \
    -C cluster0.sve.has_sme=1 \
    -C cluster0.sve.has_sve2=1 \
    -C cluster1.NUM_CORES=0 \
    -C cluster1.check_memory_attributes=0 \
    -C cluster1.ecv_support_level=2 \
    -C cluster1.enhanced_pac2_level=3 \
    -C cluster1.has_16k_granule=1 \
    -C cluster1.has_amu=1 \
    -C cluster1.has_arm_v8-1=1 \
    -C cluster1.has_arm_v8-2=1 \
    -C cluster1.has_arm_v8-3=1 \
    -C cluster1.has_arm_v8-4=1 \
    -C cluster1.has_arm_v8-5=1 \
    -C cluster1.has_arm_v8-6=1 \
    -C cluster1.has_arm_v8-7=1 \
    -C cluster1.has_arm_v8-8=1 \
    -C cluster1.has_arm_v8-9=1 \
    -C cluster1.has_arm_v9-0=1 \
    -C cluster1.has_arm_v9-1=1 \
    -C cluster1.has_arm_v9-2=1 \
    -C cluster1.has_arm_v9-3=1 \
    -C cluster1.has_arm_v9-4=1 \
    -C cluster1.has_branch_target_exception=1 \
    -C cluster1.has_brbe=1 \
    -C cluster1.has_brbe_v1p1=1 \
    -C cluster1.has_const_pac=1 \
    -C cluster1.has_gcs=1 \
    -C cluster1.has_hpmn0=1 \
    -C cluster1.has_large_system_ext=1 \
    -C cluster1.has_large_va=1 \
    -C cluster1.has_permission_indirection_s1=1 \
    -C cluster1.has_permission_indirection_s2=1 \
    -C cluster1.has_permission_overlay_s1=1 \
    -C cluster1.has_permission_overlay_s2=1 \
    -C cluster1.has_rndr=1 \
    -C cluster1.has_sve=1 \
    -C cluster1.max_32bit_el=0 \
    -C cluster1.pmb_idr_external_abort=1 \
    -C cluster1.sve.has_sme2=1 \
    -C cluster1.sve.has_sme=1 \
    -C cluster1.sve.has_sve2=1 \
    -C gic_distributor.has_nmi=1 \
    -C bp.terminal_0.quiet=1 \
    -C bp.terminal_1.quiet=1 \
    -C bp.terminal_2.quiet=1 \
    -C bp.terminal_3.quiet=1 \
    $TARMAC_TRACE \
    -C cluster0.cpu0.CONFIG64=1 \
    --application cluster0.cpu0="$PROG" \
    -C cluster0.cpu0.semihosting-cmd_line="$PROG $PROG_ARGS"
