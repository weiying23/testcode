#ifndef ALIGN_H
#define ALIGN_H 

// Calculate the size in aligned fashion.  
template<class ElementType>
// vlen: block size (in bytes) to be aligned with.
// n: total number of elements to be allocated.
// this->nnz = align<ValueType>(64, elements.size());//以64字节对齐
int align(int vlen, int n) {
  int align_size = vlen/sizeof(ElementType);  //64/sizeof(float)=64/4=16, 一次可以处理16个单精度, 双精度等于8
  return (n + align_size - 1)&(~(align_size - 1));
}

#endif /* ALIGN_H */
