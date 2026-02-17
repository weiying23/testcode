   .global _sme_support
   .global _sme_support_8
   .global _sme_support_mopa
   .align 6
_sme_support: //加载16个float，做reduce
    smstart sm
    ptrues p0.s, all
    ld1w {z0.s}, p0/z, [x1]
    faddv s0, p0, z0.s  // 32 位float归约
    str s0, [x0]
    cntw x0, all  //返回当前配置下可用的 32 位谓词寄存器数量
    smstop sm
    ret

_sme_support_8: //加载8个float，做reduce
    smstart sm
    mov x2, #8           // 设置元素数量 = 8
    ptrue p0.s           // 为单精度元素创建全真谓词 (p0)
    whilelt p1.s, xzr, x2 // 创建前8个元素的谓词 (p1)
    ld1w {z0.s}, p1/z, [x1] // 加载8个单精度数据到 z0
    faddv s0, p0, z0.s  // 32 位float归约
    str s0, [x0]
    cntw x0, all  //返回当前配置下可用的 32 位谓词寄存器数量
    smstop sm
    ret

_sme_support_mopa:
    smstart SM
    smstart ZA
    mov w14, #0           // 初始化索引寄存器
    
    ptrue p1.s
    ld1w {z0.s}, p1/z, [x1]
    ld1w {z1.s}, p1/z, [x1]
    fmopa za0.s, p1/m, p1/m, z0.s, z1.s
    
    // 使用正确的立即数范围 [0, 3]
    mov z2.s, p1/m, za0h.s[w14, #2]   // 立即数必须在 0-3 范围内
    
    str z2, [x0]
    smstop SM
    smstop ZA
    ret
