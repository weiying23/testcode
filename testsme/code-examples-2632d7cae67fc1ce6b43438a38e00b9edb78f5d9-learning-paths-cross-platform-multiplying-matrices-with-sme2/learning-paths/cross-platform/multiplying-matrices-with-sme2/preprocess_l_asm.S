// SPDX-FileCopyrightText: Copyright 2024,2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifdef __APPLE__
    .section	__TEXT,__text,regular,pure_instructions
    .global _preprocess_l_asm

    _preprocess_l_asm:
#else
    .text
    .global preprocess_l_asm
    .type preprocess_l_asm, "function"

    preprocess_l_asm:
#endif

    // preprocess_l_asm(uint64_t nbr, uint64_t nbc, const float * restrict a, float * restrict a_mod);
    // x0 : nbr
    // x1 : nbc
    // x2 : &a
    // x3 : &a_mod
    // x4 : SVLs (=cntw)
    // x5 : Exit condition for inner loop
    // x6 : a_ptr
    // x7 : Outer loop counter
    // x8 : a_base
    // x9 : a_mod store base address
    // x10: 32b Tile0 store end pointer
    // x11: SVLs*nbc
    // x12: unused
    // x13: 32b Tile1 store end pointer
    // x14: 2*nbc
    // x15: 3*nbc
    // x16: 32b tile size

// Assumptions:
// nbr in matLeft (M): any
// nbc in matLeft, nbr in matRight (K): any K > 2
// nbc in matRight (N): any
//
// Left matrix re-arrangement:
// Block of SVLs rows is transposed and contiguously stored.
// Then the same transformation is applied to remaining blocks of SVLs rows.
// The last block of rows is zero-padded to SVLs rows, if applicable.

    .cfi_startproc

// constants
    cntw    x4                      // SVLs
    mul     x11, x4, x1             // SVLs*nbc
    lsl     x14, x1, #1             // 2*nbc
    add     x15, x14, x1            // 3*nbc

    mul     x16, x4, x4             // SVLs*SVLs

    mov     x7, #0
    whilelt p0.s, x7, x0            // Tile predicate (M dimension)

.Loop_outer:
    mov     x8, x2                  // a load base address
    mov     x9, x3                  // a_mod store base address
    add     x5,  x2, x1, lsl #2     // Exit condition for inner loop

    add     x10, x9 , x11, lsl #2   // 32b Tile0 store predicate condition
    sub     x13, x10, x16, lsl #2   // 32b Tile1 store predicate condition
    whilelt pn8.b, x8, x5, vlx2     // Tile predicate-as-counter (K dimension)

.Loop_inner:
    mov     x6, x8                  // a_ptr

    mov     w12, #0                 // Load_loop counter

.Load_loop:
    psel    pn10, pn8, p0.s[w12, 0]
    psel    pn11, pn8, p0.s[w12, 1]
    psel    pn12, pn8, p0.s[w12, 2]
    psel    pn13, pn8, p0.s[w12, 3]
    ld1w    {z20.s, z28.s}, pn10/z, [x6]                // Load 2 row vectors from a_ptr
    ld1w    {z21.s, z29.s}, pn11/z, [x6, x1,  lsl #2]   // Load " "   "       from a_ptr + nbc
    ld1w    {z22.s, z30.s}, pn12/z, [x6, x14, lsl #2]   // Load " "   "       from a_ptr + nbc*2
    ld1w    {z23.s, z31.s}, pn13/z, [x6, x15, lsl #2]   // Load " "   "       from a_ptr + nbc*3
    mova    za0h.s[w12, 0:3], {z20.s-z23.s}
    mova    za1h.s[w12, 0:3], {z28.s-z31.s}

    add     x6, x6, x1, lsl #4      // a_ptr+=4*nbc FP32 elms [Bytes]
    add     w12, w12, #4            // increment counter
    cmp     w12, w4
    b.mi    .Load_loop

    mov     w12, #0                 // Store_loop counter

.Store_loop:
    whilelt pn10.b, x9, x10, vlx4
    whilelt pn11.b, x9, x13, vlx4
    mova    {z0.s-z3.s}, za0v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1v.s[w12, 0:3]
    st1w    {z0.s-z3.s}, pn10, [x9]               // Store 4 col vectors to a_mod
    st1w    {z4.s-z7.s}, pn11, [x9, x16, lsl #2]  // Store 4 col vectors to a_mod + SVLs*SVLs
    addvl   x9, x9, #4              // a_mod += 4*SVLb [Bytes]
    add     w12, w12, #4            // increment counter
    cmp     w12, w4
    b.mi    .Store_loop

    add     x9, x9, x16, lsl #2
    addvl   x8, x8, #2              // a_base += 2*SVLb [Bytes]
    whilelt pn8.b, x8, x5, vlx2
    b.first .Loop_inner

    add     x3, x3, x11, lsl #2     // &a_mod += SVLs*nbc FP32 elms [Bytes]
    add     x2, x2, x11, lsl #2     // &a += SVLs*nbc FP32 elms [Bytes]
    incw    x7

    whilelt p0.s, x7, x0
    b.first .Loop_outer

    ret

    .cfi_endproc
#ifndef __APPLE__
    .size   preprocess_l_asm, .-preprocess_l_asm
#endif
