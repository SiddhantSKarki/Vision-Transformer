// Copyright Siddhant S. Karki @ 2025

/* 
    This Project follows the design patterns used by LLM.c
    which can be found on https://github.com/karpathy/llm.c/tree/master.
    In this project, we introduce you ViT.c, a low level image detection model
    which attempts to reprocduce the model in the paper "An Image is worth 16x16 words:
    Transformers for Image Recognition at Scale".
    This project is being built to encourage the idea of using langauges like C,
    in a very constrained manner to achieve optimal minimum for a DNN faster
    than the implementations done in python.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
// #ifdef OMP
// #include <omp.h>
// #endif

// ----------------------------------------------------------------------------
// Utility Macros
#define mallocCheck(ptr, size) \
    ptr = malloc(size); \
    if (!ptr) { \
        fprintf(stderr, "malloc failed at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

#define fopenCheck(file, path, mode) \
    file = fopen(path, mode); \
    if (!file) { \
        fprintf(stderr, "Cannot open %s\n", path); \
        exit(EXIT_FAILURE); \
    }

#define memPrint(ptr, size) \
    for (size_t i = 0; i < size; i++) {\
        printf("%f ", ptr[i]); \
    } \
    printf("\n");


// ----------------------------------------------------------------------------
// Core Components
typedef struct {
    int image_size;    // Input image size (assumed square)
    int patch_size;    // Size of each patch (square)
    int num_layers;    // Number of transformer layers
    int num_heads;     // Number of attention heads
    int hidden_dim;    // Embedding dimension
    int num_classes;   // Output classes
} ViTConfig;

typedef struct {
    // Patch embedding parameters
    float* patch_proj_weight;  // [C*P*P, D]
    float* patch_proj_bias;    // [D]
    
    // Positional embeddings
    float* pos_emb;            // [N+1, D] (N = num_patches)
    
    // Transformer parameters
    float* attn_qkv_weight;    // [D, 3*D]
    float* attn_qkv_bias;      // [3*D]
    float* attn_proj_weight;   // [D, D]
    float* attn_proj_bias;     // [D]
    
    // MLP parameters
    float* mlp_fc1_weight;     // [D, 4*D]
    float* mlp_fc1_bias;       // [4*D]
    float* mlp_fc2_weight;     // [4*D, D]
    float* mlp_fc2_bias;       // [D]
    
    // Classification head
    float* head_weight;        // [D, num_classes]
    float* head_bias;          // [num_classes]
} ViTParameters;

typedef struct {
    ViTConfig config;
    ViTParameters params;
    float* params_memory;  // Contiguous memory block for all parameters
} ViT;






// ----------------------------------------------------------------------------
// Model Initialization

void vit_init(ViT* model, ViTConfig config) {
    // Calculate parameter sizes
    int P = config.patch_size;
    int C = 3;  // RGB channels
    int N = (config.image_size / P) * (config.image_size / P);
    int D = config.hidden_dim;
    
    size_t param_sizes[] = {
        C*P*P*D,        // patch_proj_weight
        D,              // patch_proj_bias
        (N+1)*D,        // pos_emb
        D*3*D,          // attn_qkv_weight
        3*D,            // attn_qkv_bias
        D*D,            // attn_proj_weight
        D,              // attn_proj_bias
        D*4*D,          // mlp_fc1_weight
        4*D,            // mlp_fc1_bias
        4*D*D,          // mlp_fc2_weight
        D,              // mlp_fc2_bias
        D*config.num_classes,  // head_weight
        config.num_classes     // head_bias
    };
    
    // Allocate contiguous memory
    size_t total_size = 0;
    for (int i = 0; i < sizeof(param_sizes)/sizeof(size_t); i++) {
        total_size += param_sizes[i];
    }
    
    mallocCheck(model->params_memory, total_size * sizeof(float));
    float* ptr = model->params_memory;
    
    // Assign parameter pointers
    model->params.patch_proj_weight = ptr;
    ptr += C*P*P*D;

    model->params.patch_proj_bias = ptr;
    ptr += D;

    model->params.pos_emb = ptr;            
    ptr += (N+1)*D;

    model->params.attn_qkv_weight = ptr;    
    ptr += D*3*D;

    model->params.attn_qkv_bias = ptr;      
    ptr += 3*D;

    model->params.attn_proj_weight = ptr;   
    ptr += D*D;

    model->params.attn_proj_bias = ptr;     
    ptr += D;

    model->params.mlp_fc1_weight = ptr;     
    ptr += D*4*D;

    model->params.mlp_fc1_bias = ptr;       
    ptr += 4*D;

    model->params.mlp_fc2_weight = ptr;     
    ptr += 4*D*D;

    model->params.mlp_fc2_bias = ptr;       
    ptr += D;

    model->params.head_weight = ptr;        
    ptr += D*config.num_classes;
    model->params.head_bias = ptr;

    // Initialize parameters (simplified for MVP)
    // TODO: Look into initialization schemes (Gaussain )
    // In real implementation, use proper initialization schemes 
    memset(model->params_memory, 0, total_size * sizeof(float));
    memPrint(model->params_memory, total_size);
}

 
// ----------------------------------------------------------------------------
// Main Program

int main() {
    // Initialize model configuration
    ViTConfig config = {
        .image_size = 4,
        .patch_size = 1,
        .num_layers = 1,
        .num_heads = 2,
        .hidden_dim = 8,
        .num_classes = 2
    };

    // Model definition
    ViT model;
    vit_init(&model, config);

    free(model.params_memory);

    return 0;
}
