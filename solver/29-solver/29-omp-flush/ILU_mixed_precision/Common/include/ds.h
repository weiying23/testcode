#ifndef DS_H
#define DS_H 

#include <iostream>
#include <tuple>
#include "align.h"
using namespace std;

// Struct of arrays for moldyn 3D data.
template <class ValueType>
class ThreeDSoa {
 public:
  int num_nodes;
  ValueType* x;
  ValueType* y;
  ValueType* z;

  ThreeDSoa (int num_nodes) {
    this->num_nodes = num_nodes;
    x = (ValueType*)_mm_malloc(sizeof(ValueType)*num_nodes, 64);  
    y = (ValueType*)_mm_malloc(sizeof(ValueType)*num_nodes, 64);  
    z = (ValueType*)_mm_malloc(sizeof(ValueType)*num_nodes, 64);  
  } 

  void SetXyz(const vector<tuple<ValueType, ValueType, ValueType> >& xyz) {
    for (int i = 0; i < num_nodes; ++i) {
      x[i] = get<0>(xyz[i]);
      y[i] = get<1>(xyz[i]);
      z[i] = get<2>(xyz[i]);
    }
  }

  ~ThreeDSoa() {
    _mm_free(x);
    _mm_free(y);
    _mm_free(z);
  }
};

// This class is used for storing the padded nnzs continuously.
template <class ValueType>
class PaddedNnz {
 public:
  int nnz;
  int* rows;
  int* cols;
  ValueType* vals;
  ValueType* facevals;

  // nnz is the total nnz after padding, which means,
  // it is a multiple of 16.
  PaddedNnz(int nnz) {
    this->nnz = nnz; 
    rows = (int*)_mm_malloc(sizeof(int)*nnz, 64); 
    cols = (int*)_mm_malloc(sizeof(int)*nnz, 64); 
    vals = (ValueType*)_mm_malloc(sizeof(ValueType)*nnz, 64); 
    facevals = (ValueType*)_mm_malloc(sizeof(ValueType) * nnz, 64);
  }
	
  ~PaddedNnz() {
    _mm_free(rows);
    _mm_free(cols);
    _mm_free(vals);
    _mm_free(facevals);
  }
};

// This class is used as containers for tiled sparse
// matrices.
template <class ValueType>
class Coo {
 public:
	int nnz;
  int actual_nnz;

	// Dimensional size of the tile.
	int width;

	// The following three items should be aligned.
	int* rows;
	int* cols;
	ValueType* vals;
    ValueType* facevals;

	Coo(const vector<tuple<int, int, ValueType> >& elements) {
		// Allocate aligned spaces.
        this->actual_nnz = elements.size();
		this->nnz = align<ValueType>(64, elements.size());//ÒÔ64×Ö½Ú¶ÔÆë
		rows = (int*)_mm_malloc(sizeof(int)*nnz, 64);
		cols = (int*)_mm_malloc(sizeof(int)*nnz, 64);
		vals = (ValueType*)_mm_malloc(sizeof(ValueType)*nnz, 64);
        facevals = (ValueType*)_mm_malloc(sizeof(int) * nnz, 64);
		// Copy elements.
		for (int i = 0; i < elements.size(); ++i) {
			rows[i] = get<0>(elements[i]);
			cols[i] = get<1>(elements[i]);
			vals[i] = get<2>(elements[i]);
            facevals[i] = 1;
		}
		// Padding.
		for (int i = elements.size(); i < nnz; ++i) {
			rows[i] = rows[0];
			cols[i] = cols[0];
			vals[i] = vals[0];
            facevals[i] = 0;   //for padding, add zero to the final results, means to do nothing for the rows[0] and cols[0] cell
		}
	}
/*
  void Refill(const vector<tuple<int, int, ValueType> >& elements) {
		_mm_free(rows);
		_mm_free(cols);
		_mm_free(vals);

      this->actual_nnz = elements.size();
	  this->nnz = align<ValueType>(64, elements.size());
	  rows = (int*)_mm_malloc(sizeof(int)*nnz, 64);
	  cols = (int*)_mm_malloc(sizeof(int)*nnz, 64);
	  vals = (ValueType*)_mm_malloc(sizeof(ValueType)*nnz, 64);
	  // Copy elements.
	  for (int i = 0; i < elements.size(); ++i) {
	  	rows[i] = get<0>(elements[i]);
	  	cols[i] = get<1>(elements[i]);
	  	vals[i] = get<2>(elements[i]);
	  }
	  // Padding.
	  for (int i = elements.size(); i < nnz; ++i) {
	  	rows[i] = rows[0];
	  	cols[i] = cols[0];
	  	vals[i] = vals[0];    
	  }
  }

	void TransferOwnership() {
	  rows = 0;
	  cols = 0;
	  vals = 0;
	}
*/
  void Free() {
    if (rows) {
		  _mm_free(rows);
      rows = 0;
    }
    if (cols) {
		  _mm_free(cols);
      cols = 0;
    }
    if (vals) {
		  _mm_free(vals);
      vals = 0;
    }
    if (facevals) {
        _mm_free(facevals);
        facevals = 0;
    }
  }

	~Coo() {
    /*
    cout << "rows: " << rows << endl;
    if (rows) {
		  _mm_free(rows);
      rows = 0;
    }
    cout << "cols: " << cols << endl;
    if (cols) {
		  _mm_free(cols);
      cols = 0;
    }
    cout << "vals: " << vals << endl;
    if (vals) {
		  _mm_free(vals);
      vals = 0;
    }
    */
	}
};

// Operator to output the value of a Coo tile.
template <class ValueType>
ostream& operator<<(ostream& os, const Coo<ValueType>& tile) {
	for (int i = 0; i < tile.nnz; ++i) {
		os << "(" << tile.rows[i] << ", " << tile.cols[i] << ", " << tile.vals[i] << ") ";
	}
	return os;
}

// Output a tile.
ofstream& operator<<(ofstream& os, const Coo<int>& tile) {
    ostringstream oss;
    for (int i = 0; i < tile.nnz; ++i) {
        oss << tile.rows[i]
            << " "
            << tile.cols[i]
            << " "
            << tile.vals[i]
            << " ";
    }
    os << oss.str();
    return os;
}

ostringstream& operator<<(ostringstream& os, const Coo<int>& tile) {
    for (int i = 0; i < tile.nnz; ++i) {
        os << tile.rows[i]
            << " "
            << tile.cols[i]
            << " "
            << tile.vals[i]
            << " "
            << tile.facevals[i]
            << " ";
    }
    return os;
}

// Output multi-thread packs.
ofstream& operator<<(ofstream& os, const vector<Coo<int> >& tiles) {
    ostringstream oss;
    for (int i = 0; i < tiles.size(); ++i) {
        oss << tiles[i];
    }
    os << oss.str();
    return os;
}

// Output three level nnzs to an output stream.
ofstream& operator<<(ofstream& os, const vector<vector<vector<Coo<int> > > >& tiles) {
    for (int i = 0; i < tiles.size(); ++i) {
        for (int j = 0; j < tiles[i].size(); ++j) {
            os << tiles[i][j];
            os << endl;  // Each tile (before grouping) is stored in one row in the file.
        }
    }
    return os;
}

// Output two level nnzs to an output stream.
ofstream& operator<<(ofstream& os, const vector<vector<Coo<int> > >& tiles) {
    int count = 0;
    for (int i = 0; i < tiles.size(); ++i) {
        for (int j = 0; j < tiles[i].size(); ++j) {
            for (int k = 0; k < tiles[i][j].actual_nnz; ++k) {
                os << tiles[i][j].rows[k] << " " << tiles[i][j].cols[k] << " " << tiles[i][j].vals[k] << " ";
                ++count;
            }
            os << endl;  // Each tile (before grouping) is stored in one row in the file.
        }
    }
    cout << "Total actual nnz written: " << count << endl;
    return os;
}

void WriteOffsets(const vector<vector<vector<Coo<int> > > >& tiles, ofstream& os) {
    // Offset for each thread.
    int total = 0;
    ostringstream oss;
    for (int i = 0; i < tiles.size(); ++i) {
        // Offset for each parallel unit.
        for (int j = 0; j < tiles[i].size(); ++j) {
            oss << total << " ";
            for (int k = 0; k < tiles[i][j].size(); ++k) {
                total += tiles[i][j][k].nnz;
            }
        }
        oss << endl;
    }
    os << total << endl;
    os << oss.str();
}

#endif /* DS_H */
