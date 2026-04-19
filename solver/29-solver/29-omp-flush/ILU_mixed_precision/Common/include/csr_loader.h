// This is a tool for loading a sparse matrix from
// Matrix Market format to a CSR format.

#ifndef CSR_LOADER_H
#define CSR_LOADER_H 

#include "align.h"
#include "csr.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

template<class ValueType>
struct CsrLoader {
  // static
  static MyCsr<ValueType>* Load(int *temf2c, int temnTFace, int temnTCell, int temnBFace) {

    // Get size info.
    int num_rows, num_cols, nnz;
    num_rows = temnTCell;
    num_cols = temnTCell;
    nnz = temnTFace;

    cout << "Num of rows: " << num_rows << endl;

    // Allocate buffers.
    vector<vector<int> > cols(num_rows);
    vector<vector<ValueType> > vals(num_rows);

    //c1 for rows, c2 for cols
    for (int i = 0; i < temnTFace; i++) {
        int c1, c2;
        c1 = temf2c[2 * i];
        c2 = temf2c[2 * i + 1];
        ValueType v = i + temnBFace;
        cols[c1].push_back(c2);
        vals[c1].push_back(v);
    }

    // Create the object from buffers.
    MyCsr<ValueType>* csr = new MyCsr<ValueType>(num_rows, num_cols, nnz);

    csr->rows[0] = 0;
    for (int i = 1; i < num_rows; ++i) {
      csr->rows[i] = csr->rows[i-1] + cols[i-1].size(); 
    }
    csr->rows[num_rows] = nnz; //the last value of rows, representing the total number of non-zero values

    int count = 0;
    for (int i = 0; i < num_rows; ++i) {
      for (int j = 0; j < cols[i].size(); ++j) {
        csr->cols[count] = cols[i][j];
        csr->vals[count] = vals[i][j];
        ++count;
      } 
    }
    //f2c to csr format
    return csr;
  } 
};

#endif /* CSR_LOADER_H */
