# SME GEMM optimization log (single-core P-core only)

Note: all numbers are single-core. Tool: single_bench (matrix_multiply_single, num_threads=1, QOS_USER_INTERACTIVE, VECLIB_MAXIMUM_THREADS=1).
Thermal state matters: pure SME fmopa peak = 265 GFLOPS hot / 496 GFLOPS cool.

## Step 0 - baseline O1, Mc=64 Kc=256 Nc=96
cool state:
- 512^3: SME 87-98, Apple BLAS 419-422
- 1024^3: SME ~80
- 2048^3: SME ~81

## Phase 1 - parameter scan (single-core 512^3)
- Nc=192: 90-95, no gain
- Kc=512: 85.9, no gain
- Mc=128: 96.7-102.0; 1024^3 87.8; 2048^3 87.3
- Mc=128 Nc=192: 97-108; 1024^3 91.9; 2048^3 93.4
- Mc=128 Kc=512 Nc=192: 98-110; 1024^3 99.1; 2048^3 99.6 -> WINNER

## Phase 2 - loop reorder kc->jc->ic
- 512^3: 107.8-113.1
- 1024^3: 102.8
- 2048^3: 102.5
-> kept

## Phase 3a - unroll inner t-loop
No clear gain. Reverted to Phase 2 k-loop.

## Phase 3 - full asm k-loop (full-tile fast path, GAS .macro T_STEP)
- 256^3: 89.5 (Phase 2 was 76.5)
- 512^3: 113.7-117.3
- 1024^3: 99.8-101.4
- 2048^3: 100.9-101.2
Finding: k-loop C++ control was NOT the main bottleneck at large sizes.
Correctness: all PASS. Synced to matrix_multiply.cpp and matrix_multiply_single.cpp. backup_phase3 created.
Current Phase 3 table (single-core, fresh run):
- 128^3: 66.0
- 256^3: 76.5-90.5 (variance)
- 512^3: 112.7
- 1024^3: 101.2
- 2048^3: 101.1
Phase 4 2-stage software pipeline (double buffer z3-z8 / z9-z14):
- 512^3: 109.1-114.6 (vs Phase3 ~113-117) -> neutral
- 1024^3: ~101.3
- 2048^3: ~100.6
- Kc=256 L1-tile attempt: 512^3 101.3, 1024^3 92.2, 2048^3 93.1 -> worse, reverted Kc=512
Current state: pipelined asm Kc=512 (not yet synced to main).
Phase 4b 4-stage software pipeline (24 z regs, distance 3 t-steps):
- 512^3: 101.7-112.1
- 1024^3: ~100.5-101.9
- 2048^3: ~100.8-101.2
Same as 2-stage and non-pipelined. Implication: B-load latency is not the limiter, or OoO already hides it. Bottleneck likely in i/j C++ loops + C tile I/O + packing.
