// SPDX-FileCopyrightText: Copyright 2024,2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifdef __APPLE__
    .section	__TEXT,__text,regular,pure_instructions
    .global _matmul_asm_impl

    _matmul_asm_impl:
#else
    .text
    .global matmul_asm_impl
    .type matmul_asm_impl, "function"

    matmul_asm_impl:
#endif

    // matmul_asm_impl(uint64_t M, uint64_t K, uint64_t N, const float *restrict matLeft_mod, const float *restrict matRight, float *restrict matResult)
    // x0 : M
    // x1 : K, lda
    // x2 : N, ldc, ldb
    // x3 : &matLeft_mod
    // x4 : &matRight
    // x5 : &matResult_opt
    // x6 : SVLs-2
    // x7 : a_ptr pointer
    // x8 : a_ptr end address
    // x9 : c_base pointer
    // x10: c_ptr0 pointer
    // x11: Exit condition for N loop
    // x12: M loop counter
    // x13: Store loop counter
    // x14: Predicate select index
    // x15: Exit condition for K loop
    // x16: b_base pointer
    // x17: b_ptr pointer
    // x18: *don't use* (platform register)
    // x19: ldb + SVLs
    // x20: SVLs*lda + SVLs
    // x21: c_ptr1 pointer
    // x22: SVLs*lda
    // x23: SVLs*ldc
    // x24: (SVLs+1)*ldc

// Assumptions:
// nbr in matLeft (M): any
// nbc in matLeft, nbr in matRight (K): any K > 2
// nbc in matRight (N): any
//
// Left matrix is pre-arranged.
//
// 32-bit accumulator mapping with 2x2 tiles processing

    .cfi_startproc

    stp     x19, x20, [sp, #-48]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]

// constants
    cntw    x6                      // SVLs
    mul     x22, x6, x1             // SVLs*lda
    mul     x23, x6, x2             // SVLs*ldc
    add     x24, x23, x2            // SVLs*ldc + ldc
    add     x11, x4, x2, lsl #2     // Exit condition for N loop
    mov     x12, #0
    cntb    x6                      // SVLb
    mov     x14, #0
    ptrue   pn10.b                  // Predicate as counter for SME2 VLx2 (a_ptr loads)
    whilelt pn8.s, x12, x0, vlx2    // tiles predicate (M dimension)
    sub     w6, w6, #8              // SVLb-8

.Loop_M:
    // Extracting tile 0/1 and tile 2/3 predicates (M dimension) from vlx2 predicate.
    pext    { p2.s, p3.s }, pn8[0]
    mov     x16, x4                 // b_base
    mov     x9, x5                  // c_base

    whilelt pn9.b, x16, x11, vlx2   // tiles predicate (N dimension)

.Loop_N:
    mov     x7, x3                  // a_ptr = a_base
    mov     x17, x16                // b_ptr = b_base
    mov     x10, x9                 // c_ptr0 = c_base

    // Extracting tile 0/2 and tile 1/3 predicates (N dimension) from vlx2 predicate.
    pext    { p0.b, p1.b }, pn9[0]

    add     x8, x3, x22, lsl #2     // a_base + SVLs*lda FP32 elms [Bytes]
    addvl   x15, x8, #-1            // Exit condition for K loop
    ld1w    {z1.s},  p2/z,   [x7]   // Load 1st vector from a_ptr

    zero    {za}
    ld1w    {z2.s-z3.s},  pn9/z,   [x17]  // Load 2 vectors from b_ptr

    fmopa   za0.s,  p2/m,   p0/m,   z1.s,   z2.s  // ZA0 += 1st a_ptr vector OP 1st b_ptr vector
    ld1w    {z5.s},  p3/z,   [x7, x22, lsl #2]    // Load 2nd vector from a_ptr
    addvl   x7, x7, #1                            // a_ptr += SVLb [Bytes]

.Loop_K:
    fmopa   za2.s,  p3/m,   p0/m,   z5.s,   z2.s       // ZA2 += 2nd a_ptr vector OP 1st b_ptr vector

    fmopa   za1.s,  p2/m,   p1/m,   z1.s,   z3.s       // ZA1 += 1st a_ptr vector OP 2nd b_ptr vector
    ld1w    {z0.s-z1.s},  pn10/z,   [x7]               // Load next 2 vectors from a_ptr

    fmopa   za3.s,  p3/m,   p1/m,   z5.s,   z3.s       // ZA3 += 2nd a_ptr vector OP 2nd b_ptr vector
    ld1w    {z6.s-z7.s},  pn9/z,   [x17, x2, lsl #2]   // Load next 2 vectors from b_ptr

    fmopa   za0.s,  p2/m,   p0/m,   z0.s,   z6.s       // ZA0 += 1st a_ptr vector OP 1st b_ptr vector
    psel    pn11, pn10, p3.s[w14, 0]                   // Select predicate-as-counter
    ld1w    {z4.s-z5.s},  pn11/z,   [x7, x22, lsl #2]  // Load next 2 vectors from a_ptr

    fmopa   za2.s,  p3/m,   p0/m,   z4.s,   z6.s       // ZA2 += 2nd a_ptr vector OP 1st b_ptr vector
    add     x17, x17, x2, lsl #3                       // b_ptr += 2*ldb FP32 elms [Bytes]

    fmopa   za1.s,  p2/m,   p1/m,   z0.s,   z7.s       // ZA1 += 1st a_ptr vector OP 2nd b_ptr vector

    fmopa   za3.s,  p3/m,   p1/m,   z4.s,   z7.s       // ZA3 += 2nd a_ptr vector OP 2nd b_ptr vector
    ld1w    {z2.s-z3.s},  pn9/z,   [x17]               // Load next 2 vectors from b_ptr

    fmopa   za0.s,  p2/m,   p0/m,   z1.s,   z2.s       // ZA0 += 1st a_ptr vector OP 1st b_ptr vector
    addvl   x7, x7, #2                                 // a_ptr += 2*SVLb [Bytes]

    cmp     x7, x15
    b.mi    .Loop_K

    fmopa   za2.s,  p3/m,   p0/m,   z5.s,   z2.s       // ZA2 += 2nd a_ptr vector OP 1st b_ptr vector

    fmopa   za1.s,  p2/m,   p1/m,   z1.s,   z3.s       // ZA1 += 1st a_ptr vector OP 2nd b_ptr vector

    fmopa   za3.s,  p3/m,   p1/m,   z5.s,   z3.s       // ZA3 += 2nd a_ptr vector OP 2nd b_ptr vector
    add     x17, x17, x2, lsl #2                       // b_ptr += 2*ldb FP32 elms [Bytes]

    cmp     x7, x8
    b.pl    .Ktail_end

.Ktail_start:
    ld1w    {z1.s},  p2/z,   [x7]
    ld1w    {z2.s-z3.s},  pn9/z,   [x17]

    fmopa   za0.s,  p2/m,   p0/m,   z1.s,   z2.s
    ld1w    {z5.s},  p3/z,   [x7, x22, lsl #2]

    fmopa   za2.s,  p3/m,   p0/m,   z5.s,   z2.s

    fmopa   za1.s,  p2/m,   p1/m,   z1.s,   z3.s

    fmopa   za3.s,  p3/m,   p1/m,   z5.s,   z3.s

.Ktail_end:
    mov     w13, #0
    psel    pn11, pn9, p2.b[w13, 0]
    psel    pn12, pn9, p3.b[w13, 0]
    // Move from ZA tiles to vectors: z0 = za0h[1], z1 = za1h[1], z2 = za2h[1], z3 = za3h[1]
    mova    { z0.b-z3.b }, za0h.b[w13, 0:3]
    st1w    { z0.s-z1.s }, pn11, [x10]                  // Store to c_ptr0
    st1w    { z2.s-z3.s }, pn12, [x10, x23, lsl #2]     // Store to c_ptr0 + SVLs*ldc
.Loop_store_ZA:
    psel    pn11, pn9, p2.b[w13, 4]
    psel    pn12, pn9, p3.b[w13, 4]
    mova    { z0.b-z3.b }, za0h.b[w13, 4:7]
    st1w    { z0.s-z1.s }, pn11, [x10, x2,  lsl #2]      // Store to c_ptr0 + ldc
    st1w    { z2.s-z3.s }, pn12, [x10, x24,  lsl #2]     // Store to c_ptr0 + (SVLs+1)*ldc

    add     x10, x10, x2, lsl #3    // c_ptr0 += 2*ldc FP32 elms [Bytes]
    add     w13, w13, #8

    psel    pn11, pn9, p2.b[w13, 0]
    psel    pn12, pn9, p3.b[w13, 0]
    mova    { z0.b-z3.b }, za0h.b[w13, 0:3]
    st1w    { z0.s-z1.s }, pn11, [x10]                  // Store to c_ptr0
    st1w    { z2.s-z3.s }, pn12, [x10, x23, lsl #2]     // Store to c_ptr0 + SVLs*ldc
    cmp     w13, w6
    b.mi    .Loop_store_ZA

    psel    pn11, pn9, p2.b[w13, 4]
    psel    pn12, pn9, p3.b[w13, 4]
    mova    { z0.b-z3.b }, za0h.b[w13, 4:7]
    st1w    { z0.s-z1.s }, pn11, [x10, x2,  lsl #2]      // Store to c_ptr0 + ldc
    st1w    { z2.s-z3.s }, pn12, [x10, x24,  lsl #2]     // Store to c_ptr0 + (SVLs+1)*ldc

    addvl   x9, x9, #2
    addvl   x16, x16, #2            // b_base += 2*SVLb [Bytes]
    whilelt pn9.b, x16, x11, vlx2   // tile predicate (N dimension)
    b.first .Loop_N

    add     x3, x3, x22, lsl #3     // a_base += 2*SVLs*lda FP32 elms [Bytes]
    add     x5, x5, x23, lsl #3     // c_base += 2*SVLs*ldc FP32 elms [Bytes]
    incw    x12, all, mul #2        // M loop counter += 2* SVLs
    whilelt pn8.s, x12, x0, vlx2    // tiles predicate (M dimension)
    b.first    .Loop_M

    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #48

    ret

    .cfi_endproc

#ifndef __APPLE__
    .size   matmul_asm_impl, .-matmul_asm_impl
#endif
