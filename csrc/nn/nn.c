#define NN_IMPLEMENTATION
#define UTILS_IMPLEMENTATION
#include <stdio.h>
#include "nn.h"
#include "utils.h"
#include <time.h>


typedef struct {
    Mat a0;
    Mat w1, b1, a1;
    Mat w2, b2, a2;
} Model;



Mat forward_xor(Model model, Mat input) {
    model.a0 = input; 
    
    printf("--------------Layer1 Computation---------\n");
    // First Layer Operation
    mat_mul(model.a1, model.a0, model.w1);
    mat_add(model.a1, model.a1, model.b1);
    mat_sigmoid(model.a1);

    printf("--------------Layer2 Computation---------\n");
   
    mat_mul(model.a2, model.a1, model.w2);
    mat_add(model.a2, model.a2, model.b2);
    mat_sigmoid(model.a2);
    
    return model.a2;
}


int main(void) {
    srand(time(0));
    float input[2] = {0, 1};
    float output[1] = {1};
    Mat x = {.rows=1, .cols=2, .dat=input};
    Mat y = {.rows=1, .cols=1, .dat=output};
    Model model;
     // Layer1
    model.w1 = mat_malloc(2, 2);
    model.b1 = mat_malloc(1, 2); 
    model.a1 = mat_malloc(1, 2); 
   
    // Layer2
    model.w2 = mat_malloc(2, 1);
    model.b2 = mat_malloc(1, 1);
    // Second Layer Operation
    model.a2 = mat_malloc(1, 1);
        
    printf("--------------Layer1 Init---------\n");
     
    mat_randomize(model.w1, 0, 10);
    mat_randomize(model.b1, 0, 10);
    
    MAT_DISPLAY(model.w1);
    MAT_DISPLAY(model.b1);
    
    printf("--------------Layer2 Init---------\n");

    mat_randomize(model.w2, 0, 10);
    mat_randomize(model.b2, 0, 10);
    MAT_DISPLAY(model.w2);
    MAT_DISPLAY(model.b2);
    
     
    // Printing First layer output
    MAT_DISPLAY(model.a1);
   
    // Printing Second Layer Output 
    // FINAL OUTPUT -> y_hat
    MAT_DISPLAY(model.a2);
    Mat pred = forward_xor(model, x);
    
    // Code for Backward Prop
    float loss = MSE_LOSS(y, pred);
    printf("Loss = %f\n", loss);
    

    return 0;
}
