#ifndef NN_H_
#define NN_H_

#ifndef NN_MALLOC
#define NN_MALLOC malloc
#endif

#include <stddef.h>
#include <stdlib.h>
#include <math.h>


#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif


typedef struct {
    size_t rows;
    size_t cols;
    float* dat;
} Mat;

#define MAT_AT(mat1, i, j) (mat1).dat[(i)*(mat1.cols) + (j)]
#define MAT_DISPLAY(m) mat_display(m, #m);
// Definitions HERE

Mat mat_malloc(size_t rows, size_t cols);
void mat_mul(Mat dest, Mat mat1, Mat mat2);
void mat_add(Mat dest, Mat mat1, Mat mat2);
float rand_float(void);
void mat_scale(Mat mat1, float scalar);
void mat_display(Mat mat1, const char* name);
void mat_randomize(Mat mat1, float low, float high);
void mat_fill(Mat mat1, float val);
float sigmoidf(float num);
void mat_sigmoid(Mat mat1);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION




// Implementations HERE
Mat mat_malloc(size_t rows, size_t cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.dat = NN_MALLOC(sizeof(*m.dat)*rows*cols);
    assert(m.dat != NULL);
    return m;
}

void mat_mul(Mat dest, Mat mat1, Mat mat2) {
    
    // Want to check if the matrices are multipliable
    NN_ASSERT(mat1.cols == mat2.rows); 

    // Also want to check if the destination matrix (allocated by US)
    // have enough space for the resultant things
    NN_ASSERT(dest.rows==mat1.rows);
    NN_ASSERT(dest.cols==mat2.cols);
    size_t size = mat1.rows; 
    // MAtrix Mult is basically iteration over te destinatiokn table
    for (size_t i = 0; i < dest.rows; ++i) {
        for (size_t j = 0; j < dest.cols; ++j) {
            MAT_AT(dest, i, j) = 0.f;
            for (size_t k = 0; k < size; ++k) {
                MAT_AT(dest, i, j) += MAT_AT(mat1, i, k)*MAT_AT(mat2, k, j); 
            }            
        }
    }
}

float sigmoidf(float num) {
    return 1.f / (1 + exp(-num));
}

void mat_sigmoid(Mat mat1) {
    for (size_t i = 0; i < mat1.rows; ++i) {
        for (size_t j = 0; j < mat1.cols; ++j) {
            MAT_AT(mat1, i, j) = sigmoidf(MAT_AT(mat1, i, j));
        }
    }
}

void mat_scale(Mat mat1, float val) {
    for (size_t i = 0; i < mat1.rows; ++i) {
        for (size_t j = 0; j < mat1.cols; ++j) {
            MAT_AT(mat1, i, j) *= val;
        } 
    }
}

void mat_add(Mat dest, Mat mat1, Mat mat2) {
    
    // Want to check if the matrices are multipliable
    NN_ASSERT(mat1.rows == mat2.rows); 
    NN_ASSERT(mat1.cols == mat2.cols); 

    // Also want to check if the destination matrix (allocated by US)
    // have enough space for the resultant things
    NN_ASSERT(dest.rows==mat1.rows);
    NN_ASSERT(dest.cols==mat1.cols);
    // Matrix add is basically iteration over te destinatiokn table
    for (size_t i = 0; i < dest.rows; ++i) {
        for (size_t j = 0; j < dest.cols; ++j) {
            MAT_AT(dest, i, j) = MAT_AT(mat1, i, j) + MAT_AT(mat2, i, j);
        }
    }
}

float rand_float(void) {
    return (float) rand() / (float) RAND_MAX;
}


void mat_fill(Mat mat1, float val) {
    for (size_t i = 0; i < mat1.rows; ++i) {
        for (size_t j = 0; j < mat1.cols; ++j) {
            MAT_AT(mat1, i, j) = val;
        } 
    }
}


void mat_randomize(Mat mat1, float low, float high) {
    for (size_t i = 0; i < mat1.rows; ++i) {
        for (size_t j = 0; j < mat1.cols; ++j) {
            MAT_AT(mat1, i, j) = rand_float()*(low-high) + low;
        }
    }
}

void mat_display(Mat mat1, const char* name) {
    printf("%s = [\n", name);
    for (size_t i = 0; i < mat1.rows; ++i) {
        for (size_t j = 0; j < mat1.cols; ++j) {
            printf("    %f", MAT_AT(mat1, i, j));
        }
        printf("\n");
    }
    printf("] (Shape: %ldx%ld)\n", mat1.rows, mat1.cols);
}



#endif // NN_IMPLEMENTATION
