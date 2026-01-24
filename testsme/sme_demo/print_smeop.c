#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>

typedef float float32_t;

extern int sme_support(float32_t *, const float *);
extern int sme_support_8(float32_t *, const float *);
extern void sme_support_mopa(float32_t *, const float *);

const float in[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    //{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
const float in2[8] = {1, 2, 3, 4, 5, 6, 7, 8};


int main() {

    float32_t out;
    float32_t out2;
    float32_t out3[8];
    int c;
    c = sme_support(&out, in);
    c = sme_support_8(&out2, in2);
    sme_support_mopa(out3, in);
    printf("size is %d. sum is %f\n", c, out);
    printf("size is %d. sum is %f\n", c, out2);
    printf("output is %f\n", out3[1]);    
}
