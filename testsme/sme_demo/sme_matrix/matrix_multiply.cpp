#define trans_a 1
#define trans_b 1
#include <cstdio>
#include <cstdint> 
#include <arm_sve.h>
#include <arm_sme.h>
#include <arm_acle.h>

void test_kernel(){
    __asm__ volatile(
        "smstart"
    );

    printf("In streaming_mode: %d",__arm_in_streaming_mode());

    __asm__ volatile(
        "smstop"
    );
} 


void gemmkernel(double *mata, double *matb, double *matc, int M, int N, int K, double alpha) __arm_streaming
{
    printf("set done ");
    //return;
    //__asm__ volatile("SMSTART SM" ::: "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13","d14","d15");
    //__asm__ volatile("SMSTART ZA" ::: "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13","d14","d15")
    __asm__ volatile("SMSTART ZA");

    uint64_t vscale;
    vscale = svcntd();
    svbool_t pm, pn, pk;
    svfloat64_t src1, src2, ssrc3, src4, src5;

    for(size_t i = 0; i < M; i += vscale){
        pm = svwhilelt_b64_u32(i,M);
        for (size_t j = 0; j < N; j += vscale){
            pn = svwhilelt_b64_u32(j,N);
            svzero_mask_za(1);
            for (size_t k = 0; k < K; k += vscale){
                pk = svwhilelt_b64_u32(k,K);
                for (size_t t = 0; t < vscale; t++){
                    if (i + t == M)
                        break;
                    svld1_hor_za64(1, t, pk, mata + (i + t) * K + k);
                }
                for(size_t t = 0; t < vscale; t++){
                    if (k + t == K)
                        break;
                    src1 = svread_ver_za64_f64_m(src1, pm, 1, t);
                    src2 = svld1_f64(pn, matb + (k + t) * N + j);
                    svmopa_za64_f64_m(0, pm, pn, src1, src2);
                }
            }
            for(size_t t = 0; t < vscale; t++){
                if(i + t == M)
                    break;
                svst1_hor_za64(0, t, pn, matc + (i + t) * N + j);
            }
        }
    }
    __asm__ volatile("SMSTOP ZA");

}
