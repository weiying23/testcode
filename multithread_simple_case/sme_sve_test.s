   .text
   .global _sve_support
   .align 4
_sve_support:
   smstart
   ptrue p0.b
   fmla z0.s, p0/m, z30.s, z31.s
   smstop
   ret
