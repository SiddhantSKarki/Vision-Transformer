// Copyright Siddhant S. Karki @ 2025

/* 
    This Project follows the design patterns used by LLM.c
    which can be found on https://github.com/karpathy/llm.c/tree/master.
    In this project, we introduce you ViT.c, a low level image detection model
    which attempts to create the model in the paper "An Image is worth 16x16 words:
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
#ifdef OMP
#include <omp.h>
#endif

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

#define RANDOM_STATE 69

#define __IMAGE(i_p, j_dim, image, config) ({ \
    size_t img_dims_c = config.image_size; \
    size_t img_dims_r = config.image_size * config.channels; \
    size_t p_size = config.patch_size; \
    size_t p_row = i_p / ((img_dims_c / p_size)); \
    size_t p_col = i_p % (img_dims_c / p_size); \
    size_t init_idx = (p_row * img_dims_c + p_col) * p_size; \
    size_t dim_idx = (j_dim / p_size) * img_dims_c; \
    size_t offset = j_dim % p_size; \
    size_t idx = init_idx + dim_idx + offset; \
    if (idx >= img_dims_r * img_dims_c) { \
        fprintf(stderr, "Index out of bounds at IMAGE %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
    image[idx]; \
})

#define __KERNEL(j_kernel, ker_dim, model) ({ \
    float* wei = model.params.patch_proj_weight; \
    size_t p_size = model.config.patch_size; \
    size_t weight_dims_y = model.config.hidden_dim * p_size; \
    size_t ker_row = j_kernel * p_size; \
    size_t dim_idx = (ker_dim / p_size) * weight_dims_y; \
    size_t offset = ker_dim % p_size; \
    size_t idx = ker_row + dim_idx + offset; \
    if (idx >= p_size * weight_dims_y) { \
        fprintf(stderr, "Index out of bounds at KERNEL WEIGHT %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
    wei[idx]; \
})

#define __KERNEL_BIAS(j_kernel, model) ({ \
    float* bias = model.params.patch_proj_bias; \
    size_t p_size = model.config.patch_size; \
    size_t weight_dims_y = model.config.hidden_dim; \
    size_t ker_row = j_kernel; \
    size_t idx = ker_row; \
    if (idx >= weight_dims_y) { \
        fprintf(stderr, "Index out of bounds at KERNEL BIAS %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
    bias[idx]; \
})

#define __PROJ_PATCH(i_patch, j_kernel, model, out) ({ \
    size_t cols = model.config.hidden_dim; \
    size_t idx = cols*i_patch + j_kernel; \
    &out[idx]; \
})


// #TODO: make things work for multiple channels  
#define ___CONVOLUTION(image, model, out, i_patch, j_kernel) ({ \
    size_t kernel_dims = model.config.patch_size * model.config.patch_size; \
    float result = 0.0; \
    for (size_t idx = 0; idx < kernel_dims; ++idx) { \
        result += __IMAGE(i_patch, idx, image, model.config) * __KERNEL(j_kernel, idx, model); \
    } \
    result += __KERNEL_BIAS(j_kernel, model); \
    result; \
})


// ----------------------------------------------------------------------------
// Core Components
typedef struct {
    int image_size;    // Input image size (assumed square)
    int channels; // Channels
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
// Layer Implementations

// Patch embedding: Convert image to sequence of patch embeddings
void patch_embed(float* out, const float* image, const ViT* model,
                int C, int H, int W, int P, int D) {
    int N = (H / P) * (W / P);
    int pp = P * P;

    #pragma omp parallel for collapse(3)
    for (size_t c_idx = 0; c_idx < C; ++c_idx) {
        for (int i_patch = 0; i_patch < N; ++i_patch) {
            for (int j_kernel = 0; j_kernel < D; ++j_kernel) {
                int patch_idx = c_idx*N + i_patch;
                *(__PROJ_PATCH(i_patch, j_kernel,  (*model), out)) += ___CONVOLUTION(image, (*model), out, patch_idx, j_kernel);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Forward Pass

float* vit_forward(ViT* model, const float* image) {
    ViTConfig c = model->config;
    int P = c.patch_size; // Patch Size
    int C = c.channels;  // RGB channels
    int H = c.image_size; // Height
    int W = c.image_size; // Width
    int D = c.hidden_dim; // Patch Embedding Dimension
    int N = (H/P) * (W/P); // Number of Patches

    // Allocate activation memory
    float* patches, * attn_out, * mlp_out, * logits;
    // printf("Allocating: %ld\n", N*D*sizeof(float));
    mallocCheck(patches, N*D*sizeof(float));
    // mallocCheck(attn_out, N*D*sizeof(float));

    // 1. Patch embedding
    patch_embed(patches, image,
               model,
               C, H, W, P, D);

    // 2. Add positional embeddings
    // add_position_embeddings(patches, model->params.pos_emb, 1, N, D);

    // Multi-head attention
    // attention_forward(attn_out, patches, 1, N, D, c.num_heads);

    // Cleanup intermediate buffers
    // free(patches);

    return patches;
}

// ----------------------------------------------------------------------------
// Model Allocation

void vit_alloc(ViT* model, int verbose) {
    if (verbose) {
        printf("************************************\n");
        printf("Vision Transformer (ViT) - Initialization\n");
        printf("************************************\n");
    }
    ViTConfig config = model->config;
    // Calculate parameter sizes
    int P = config.patch_size;
    int C = config.channels;  // RGB channels
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
    if (verbose) printf("Allocating memory for patch projection weights: %ld bytes\n", C*P*P*D*sizeof(float));
    model->params.patch_proj_weight = ptr;
    ptr += C*P*P*D;

    if (verbose) printf("Allocating memory for patch projection biases: %ld bytes\n", D*sizeof(float));
    model->params.patch_proj_bias = ptr;
    ptr += D;

    if (verbose) printf("Allocating memory for positional embeddings: %ld bytes\n", (N+1)*D*sizeof(float));
    model->params.pos_emb = ptr;            
    ptr += (N+1)*D;

    if (verbose) printf("====================================\n");
    if (verbose) printf("Allocating memory for attention QKV weights: %ld bytes\n", D*3*D*sizeof(float));
    model->params.attn_qkv_weight = ptr;    
    ptr += D*3*D;

    if (verbose) printf("Allocating memory for attention QKV biases: %ld bytes\n", 3*D*sizeof(float));
    model->params.attn_qkv_bias = ptr;      
    ptr += 3*D;

    if (verbose) printf("Allocating memory for attention projection weights: %ld bytes\n", D*D*sizeof(float));
    model->params.attn_proj_weight = ptr;   
    ptr += D*D;

    if (verbose) printf("Allocating memory for attention projection biases: %ld bytes\n", D*sizeof(float));
    model->params.attn_proj_bias = ptr;     
    ptr += D;

    if (verbose) printf("====================================\n");
    if (verbose) printf("Allocating memory for MLP first layer weights: %ld bytes\n", D*4*D*sizeof(float));
    model->params.mlp_fc1_weight = ptr;     
    ptr += D*4*D;

    if (verbose) printf("Allocating memory for MLP first layer biases: %ld bytes\n", 4*D*sizeof(float));
    model->params.mlp_fc1_bias = ptr;       
    ptr += 4*D;

    if (verbose) printf("Allocating memory for MLP second layer weights: %ld bytes\n", 4*D*D*sizeof(float));
    model->params.mlp_fc2_weight = ptr;     
    ptr += 4*D*D;

    if (verbose) printf("Allocating memory for MLP second layer biases: %ld bytes\n", D*sizeof(float));
    model->params.mlp_fc2_bias = ptr;       
    ptr += D;

    if (verbose) printf("====================================\n");
    if (verbose) printf("Allocating memory for classification head weights: %ld bytes\n", D*config.num_classes*sizeof(float));
    model->params.head_weight = ptr;        
    ptr += D*config.num_classes;

    if (verbose) printf("Allocating memory for classification head biases: %ld bytes\n", config.num_classes*sizeof(float));
    model->params.head_bias = ptr;

    // Testing memory allocation
    if (ptr - model->params_memory != total_size - config.num_classes) {
        fprintf(stderr, "Memory allocation error: expected %ld bytes, but allocated %ld bytes\n", total_size - config.num_classes, ptr - model->params_memory);
        exit(EXIT_FAILURE);
    } else {
        printf("************************************\n");
        printf("Memory allocation successful.\n");
        printf("************************************\n");
    }

    // Initialize parameters (simplified for MVP)
    // TODO: Look into initialization schemes (Gaussian)
    // In real implementation, use proper initialization schemes 
    memset(model->params_memory, 0.0, total_size * sizeof(float));
}


// ----------------------------------------------------------------------------
// Parameter Initialization

// Helper function to initialize an array with random values between 0 and 1
void initialize_random(float* array, size_t size) {
    for (size_t i = 0; i < size; i++) {
        array[i] = (float)rand() / (float)RAND_MAX;  // Random value between 0 and 1
    }
}

void initialize_zeros(float* array, size_t size) {
    memset(array, 0, size * sizeof(float));
}

void initialize_ones(float* array, size_t size) {
    for (size_t i = 0; i < size; i++) {
        array[i] = 1.0f;
    }
}

// Component initialization functions
void init_patch_proj_weight(float* weights, size_t size) {
    initialize_random(weights, size);
}

void init_patch_proj_bias(float* biases, size_t size) {
    initialize_random(biases, size);
}

void init_pos_emb(float* pos_emb, size_t size) {
    initialize_random(pos_emb, size);
}

void init_attn_qkv_weight(float* weights, size_t size) {
    initialize_random(weights, size);
}

void init_attn_qkv_bias(float* biases, size_t size) {
    initialize_random(biases, size);
}

void init_attn_proj_weight(float* weights, size_t size) {
    initialize_random(weights, size);
}

void init_attn_proj_bias(float* biases, size_t size) {
    initialize_random(biases, size);
}

void init_mlp_fc1_weight(float* weights, size_t size) {
    initialize_random(weights, size);
}

void init_mlp_fc1_bias(float* biases, size_t size) {
    initialize_random(biases, size);
}

void init_mlp_fc2_weight(float* weights, size_t size) {
    initialize_random(weights, size);
}

void init_mlp_fc2_bias(float* biases, size_t size) {
    initialize_random(biases, size);
}

void init_head_weight(float* weights, size_t size) {
    initialize_random(weights, size);
}

void init_head_bias(float* biases, size_t size) {
    initialize_random(biases, size);
}

// Wrapper function to initialize the entire model parameters
void vit_init(ViT* model, int verbose) {
    // Calculate parameter sizes
    ViTConfig config = model->config;
    int P = config.patch_size;
    int C = model->config.channels;  // RGB channels
    int N = (config.image_size / P) * (config.image_size / P);
    int D = config.hidden_dim;
    if (verbose) {
        printf("************************************\n");
        printf("Vision Transformer (ViT) - Random Initialization\n");
        printf("************************************\n");
    }

    // Seed the random number generator
    srand(RANDOM_STATE);

    // Initialize each component
    if (verbose) printf("Initializing patch projection weights\n");
    init_patch_proj_weight(model->params.patch_proj_weight, P * P * D);

    if (verbose) printf("Initializing patch projection biases\n");
    init_patch_proj_bias(model->params.patch_proj_bias, D);

    if (verbose) printf("Initializing positional embeddings\n");
    init_pos_emb(model->params.pos_emb, (N + 1) * D);

    if (verbose) printf("====================================\n");
    if (verbose) printf("Initializing attention QKV weights\n");
    init_attn_qkv_weight(model->params.attn_qkv_weight, D * 3 * D);

    if (verbose) printf("Initializing attention QKV biases\n");
    init_attn_qkv_bias(model->params.attn_qkv_bias, 3 * D);

    if (verbose) printf("Initializing attention projection weights\n");
    init_attn_proj_weight(model->params.attn_proj_weight, D * D);

    if (verbose) printf("Initializing attention projection biases\n");
    init_attn_proj_bias(model->params.attn_proj_bias, D);

    if (verbose) printf("====================================\n");
    if (verbose) printf("Initializing MLP first layer weights\n");
    init_mlp_fc1_weight(model->params.mlp_fc1_weight, D * 4 * D);

    if (verbose) printf("Initializing MLP first layer biases\n");
    init_mlp_fc1_bias(model->params.mlp_fc1_bias, 4 * D);

    if (verbose) printf("Initializing MLP second layer weights\n");
    init_mlp_fc2_weight(model->params.mlp_fc2_weight, 4 * D * D);

    if (verbose) printf("Initializing MLP second layer biases\n");
    init_mlp_fc2_bias(model->params.mlp_fc2_bias, D);

    if (verbose) printf("====================================\n");
    if (verbose) printf("Initializing classification head weights\n");
    init_head_weight(model->params.head_weight, D * config.num_classes);

    if (verbose) printf("Initializing classification head biases\n");
    init_head_bias(model->params.head_bias, config.num_classes);

    printf("************************************\n");
    printf("Parameter initialization successful.\n");
    printf("************************************\n");
}



// ----------------------------------------------------------------------------
// Visualization Functions
// Helper function to print a 2D matrix
void print_matrix(const float* matrix, int rows, int cols, const char* name) {
    printf("Matrix: %s (%d x %d)\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.4f ", matrix[i * cols + j]);  // Print each element with 4 decimal places
        }
        printf("\n");
    }
    printf("\n");
}

// Component-specific visualization functions
void visualize_patch_proj_weight(const ViT* model) {
    // This is importation for the convolution operation
    /* Assumption: 
        The jth (patch_size X patch_size) matrix in the column dimension
        represents the j-th convolution kernel. 
        The ith (patch_size X patch_size) matrix in the row dimension
        represents the convolution kernel in ith channel.  
    */
    int rows = model->config.patch_size;
    int cols = model->config.hidden_dim * model->config.patch_size;
    print_matrix(model->params.patch_proj_weight, rows, cols, "Patch Projection Weights");
}

void visualize_patch_proj_bias(const ViT* model) {
    // This is importation for the convolution operation
    /* Assumption: 
        Same as proj_weight (see proj_weight comment), but for every (patch_size X patch_size)
        matrix we only have 1 scalar bias.
    */
    int rows = 1;
    int cols = model->config.hidden_dim;
    print_matrix(model->params.patch_proj_bias, rows, cols, "Patch Projection Biases");
}

void visualize_pos_emb(const ViT* model) {
    int rows = (model->config.image_size / model->config.patch_size) * (model->config.image_size / model->config.patch_size) + 1;
    int cols = model->config.hidden_dim;
    print_matrix(model->params.pos_emb, rows, cols, "Positional Embeddings");
}

void visualize_attn_qkv_weight(const ViT* model) {
    int rows = model->config.hidden_dim;
    int cols = model->config.channels * model->config.hidden_dim;
    print_matrix(model->params.attn_qkv_weight, rows, cols, "Attention QKV Weights");
}

void visualize_attn_qkv_bias(const ViT* model) {
    int rows = 1;
    int cols = model->config.channels * model->config.hidden_dim;
    print_matrix(model->params.attn_qkv_bias, rows, cols, "Attention QKV Biases");
}

void visualize_attn_proj_weight(const ViT* model) {
    int rows = model->config.hidden_dim;
    int cols = model->config.hidden_dim;
    print_matrix(model->params.attn_proj_weight, rows, cols, "Attention Projection Weights");
}

void visualize_attn_proj_bias(const ViT* model) {
    int rows = 1;
    int cols = model->config.hidden_dim;
    print_matrix(model->params.attn_proj_bias, rows, cols, "Attention Projection Biases");
}

void visualize_mlp_fc1_weight(const ViT* model) {
    int rows = model->config.hidden_dim;
    int cols = 4 * model->config.hidden_dim;
    print_matrix(model->params.mlp_fc1_weight, rows, cols, "MLP First Layer Weights");
}

void visualize_mlp_fc1_bias(const ViT* model) {
    int rows = 1;
    int cols = 4 * model->config.hidden_dim;
    print_matrix(model->params.mlp_fc1_bias, rows, cols, "MLP First Layer Biases");
}

void visualize_mlp_fc2_weight(const ViT* model) {
    int rows = 4 * model->config.hidden_dim;
    int cols = model->config.hidden_dim;
    print_matrix(model->params.mlp_fc2_weight, rows, cols, "MLP Second Layer Weights");
}

void visualize_mlp_fc2_bias(const ViT* model) {
    int rows = 1;
    int cols = model->config.hidden_dim;
    print_matrix(model->params.mlp_fc2_bias, rows, cols, "MLP Second Layer Biases");
}

void visualize_head_weight(const ViT* model) {
    int rows = model->config.hidden_dim;
    int cols = model->config.num_classes;
    print_matrix(model->params.head_weight, rows, cols, "Classification Head Weights");
}

void visualize_head_bias(const ViT* model) {
    int rows = 1;
    int cols = model->config.num_classes;
    print_matrix(model->params.head_bias, rows, cols, "Classification Head Biases");
}

// Wrapper function to visualize all parameters
void visualize_vit_parameters(const ViT* model) {
    printf("************************************\n");
    printf("Visualizing ViT Model Parameters\n");
    printf("************************************\n");

    visualize_patch_proj_weight(model);
    visualize_patch_proj_bias(model);
    visualize_pos_emb(model);
    visualize_attn_qkv_weight(model);
    visualize_attn_qkv_bias(model);
    visualize_attn_proj_weight(model);
    visualize_attn_proj_bias(model);
    visualize_mlp_fc1_weight(model);
    visualize_mlp_fc1_bias(model);
    visualize_mlp_fc2_weight(model);
    visualize_mlp_fc2_bias(model);
    visualize_head_weight(model);
    visualize_head_bias(model);

    printf("************************************\n");
    printf("Visualization complete.\n");
    printf("************************************\n");
}

// ----------------------------------------------------------------------------
// Main Program

int main() {
    // Initialize model configuration
    ViTConfig config = {
        .image_size = 512,
        .channels = 1,
        .patch_size = 32,
        .num_layers = 1,
        .num_heads = 2,
        .hidden_dim = 128,
        .num_classes = 2
    };

    // Only for testing purposes
    // ---------------------------------------------------
    int P = config.patch_size; // Patch Size
    int C = config.channels;  // RGB channels
    int H = config.image_size; // Height
    int W = config.image_size; // Width
    int D = config.hidden_dim; // Patch Embedding Dimension
    int N = (H/P) * (W/P); // Number of Patches
    // ---------------------------------------------------

    // Model definition
    ViT model;
    model.config = config;
    vit_alloc(&model, 0);
    vit_init(&model, 0);

    size_t img_mem_size = config.channels * config.image_size * config.image_size * sizeof(float);
    float* image = malloc(img_mem_size);
    if (!image) {
        fprintf(stderr, "Memory allocation failed for input image\n");
        return 1;
    }
    
    initialize_ones(image, img_mem_size/sizeof(float));

    // Run forward pass
    float* patch_embds = vit_forward(&model, image);
    
    visualize_patch_proj_weight(&model);
    visualize_patch_proj_bias(&model);
    print_matrix(image, W*C, H, "Image");
    print_matrix(patch_embds, N, D, "Patch Embeddings");

    free(patch_embds);
    free(image);
    free(model.params_memory);
    return 0;
}