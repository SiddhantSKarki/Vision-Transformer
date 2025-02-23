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

// Forward declarations
void initialize_random(float* array, size_t size);
void print_matrix(const float* matrix, int rows, int cols, const char* name);

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


// ----------------------------------------------------------------------------
// Core Components
typedef struct {
    int image_size;    // Input image size (assumed square)
    int channels; // Channels
    int patch_size;    // Size of each patch (square)
    int num_layers;    // Number of transformer layers
    int num_heads;     // Number of attention heads
    int num_blocks;
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


float get_image_value(size_t i_p, size_t j_dim, const float* image, const ViTConfig* config) {
    size_t img_dims_c = config->image_size;
    size_t img_dims_r = config->image_size * config->channels;
    size_t p_size = config->patch_size;
    size_t p_row = i_p / (img_dims_c / p_size);
    size_t p_col = i_p % (img_dims_c / p_size);
    size_t init_idx = (p_row * img_dims_c + p_col) * p_size;
    size_t dim_idx = (j_dim / p_size) * img_dims_c;
    size_t offset = j_dim % p_size;
    size_t idx = init_idx + dim_idx + offset;
    if (idx >= img_dims_r * img_dims_c) {
        fprintf(stderr, "Index out of bounds at IMAGE %s:%d\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    return image[idx];
}

float get_kernel_value(size_t j_kernel, size_t ker_dim, const ViT* model) {
    float* wei = model->params.patch_proj_weight;
    size_t p_size = model->config.patch_size;
    size_t weight_dims_y = model->config.hidden_dim * p_size;
    size_t ker_row = (j_kernel / model->config.hidden_dim)*(p_size*p_size*model->config.hidden_dim) + (j_kernel % model->config.hidden_dim)*p_size;
    size_t dim_idx = (ker_dim / p_size) * weight_dims_y;
    size_t offset = ker_dim % p_size;
    size_t idx = ker_row + dim_idx + offset;
    if (idx >= model->config.channels*p_size * weight_dims_y) {
        fprintf(stderr, "Index out of bounds at KERNEL WEIGHT %s:%d\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    return wei[idx];
}

float get_kernel_bias(size_t j_kernel, const ViT* model) {
    float* bias = model->params.patch_proj_bias;
    size_t weight_dims_y = model->config.hidden_dim;
    size_t idx = j_kernel;
    if (idx >= model->config.channels*weight_dims_y) {
        fprintf(stderr, "Index %ld out of bounds at KERNEL BIAS %s:%d\n",idx, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    return bias[idx];
}

float* get_proj_patch(size_t i_patch, size_t j_kernel, const ViT* model, float* out) {
    size_t cols = model->config.hidden_dim;
    size_t idx = cols * i_patch + j_kernel;
    return &out[idx];
}

float convolution(const float* image, const ViT* model, float* out, size_t i_patch, size_t j_kernel) {
    size_t kernel_dims = model->config.patch_size * model->config.patch_size;
    float result = 0.0;
    for (size_t idx = 0; idx < kernel_dims; ++idx) {
        float i = get_image_value(i_patch, idx, image, &model->config);
        float k = get_kernel_value(j_kernel, idx, model);
        result +=  i * k;
        // printf("%f * %f + ", i, k);
    }
    float b = get_kernel_bias(j_kernel, model);
    result += b;
    // printf("%f -- ", b);
    return result;
}


void mat_mul(float* out, const float* mat1, const float* mat2, int rows1, int cols1, int cols2) {}


void query_key_value(float* out, const float* embeddings, const ViT* model, size_t head_idx, size_t block_idx) {
    int hidden_dim = model->config.hidden_dim;
    int I = model->config.image_size;
    int P = model->config.patch_size;
    int num_patches = (I/P)*(I/P);
    int num_heads = model->config.num_heads;
    int head_size = hidden_dim / num_heads;
    const float* weights = model->params.attn_qkv_weight; 

    int block_length = block_idx*num_heads*head_size*3*hidden_dim;
    int head_length = head_size*3*hidden_dim;
    float* Q = &weights[block_length + head_idx*head_length];
    float* K = weights + head_size;
    float* V = weights + 2*head_size;

    // Inititlaizing intermediate buffers for single head computation
    size_t buff_size = head_size*(num_patches+1)*sizeof(float);
    float* q_res = (float*) malloc(buff_size); // (N+1) x head_size
    float* k_res = (float*) malloc(buff_size); // (N+1) x head_size
    float* v_res = (float*) malloc(buff_size); // (N+1) x head_size
    // ((N+1)x(hidden_dim) @ (hidden_dim x head_size)) = ((N+1)xhead_size))
    mat_mul(q_res, embeddings, Q, num_patches+1, hidden_dim, head_size);
    mat_mul(k_res, embeddings, Q, num_patches+1, hidden_dim, head_size);
    mat_mul(v_res, embeddings, Q, num_patches+1, hidden_dim, head_size);

    // mat_mul(q_res, )


}




// ----------------------------------------------------------------------------
// Layer Implementations

// Patch embedding: Convert image to sequence of patch embeddings
void patch_embed(float* out, const float* image, const ViT* model,
                int C, int H, int W, int P, int D) {
    int N = (H / P) * (W / P);
    int pp = P * P;

    #pragma omp parallel for collapse(3)
    for (int i_patch = 0; i_patch < N; ++i_patch) {
        for (int j_kernel = 0; j_kernel < D; ++j_kernel) {
            for (size_t c_idx = 0; c_idx < C; ++c_idx) {
                int patch_idx = c_idx * N + i_patch;
                int kernel_idx = c_idx * D + j_kernel;
                // printf("Channel: %ld Kernel:%d, Patch: %d ", c_idx, kernel_idx, patch_idx);
                float* proj_patch = get_proj_patch(i_patch, j_kernel, model, out);
                *proj_patch += convolution(image, model, out, patch_idx, kernel_idx);
                // printf("Curr Sum: %f--- Added into %ld,%ld\n", *proj_patch, i_patch, j_kernel);
            }
        }
    }
}

// Add positional embeddings to patch embeddings
void add_position_embeddings(float* embeddings, const float* pos_emb, 
                            int B, int N, int D) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < (N + 1); i++) {
            size_t start_idx = b*(N+1)*D + i*D;
            size_t end_idx = start_idx + D;
            for (int d = start_idx; d < end_idx; d++) {
                embeddings[d] += pos_emb[d];
            }
        }
    }
}

void add_cls_token(float* concat_embd, const float* cls_token, const float* patches,
    const ViT* model, int B, int N, int D) {

    #pragma omp parallel for
    for (int b = 0; b < B; b++) {
        for (int j = 0; j < D; j++) {
            size_t idx = b*N*D + j;
            concat_embd[idx] = cls_token[idx];
        }
    }

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int i = 1; i < (N+1); i++) {
            size_t start_idx = b*N*D + i*D;
            size_t end_idx = start_idx + D;
            for (int d = start_idx; d < end_idx; d++) {
                concat_embd[d] += patches[d-D];
            }
        }
    }

}
// Needs to output attention scores
void attention_forward(float* attn_out, const float* input_embd, const ViT* model,
    int n_blocks, int n_patches, int hidden_dim, int n_heads) {
        int head_dim = hidden_dim / n_heads;
        float* qkv = model->params.attn_proj_weight;
        float* q = qkv;
        float* k = qkv + n_patches * hidden_dim;
        float* v = qkv + 2 * n_patches * hidden_dim;

        // Compute Q, K, V
        #pragma omp parallel for
        for (int i = 0; i < n_patches; i++) {
            for (int j = 0; j < hidden_dim; j++) {
                q[i * hidden_dim + j] = 0.0f;
                for (int d = 0; d < hidden_dim; d++) {
                    q[i * hidden_dim + j] += input_embd[i * hidden_dim + d] * model->params.attn_qkv_weight[d * 3 * hidden_dim + j];
                }
                q[i * hidden_dim + j] += model->params.attn_qkv_bias[j];
            }
        }

        // Compute attention scores
        float* attn_scores = (float*)malloc(n_patches * n_patches * sizeof(float));
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n_patches; i++) {
            for (int j = 0; j < n_patches; j++) {
                attn_scores[i * n_patches + j] = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    attn_scores[i * n_patches + j] += q[i * hidden_dim + d] * k[j * hidden_dim + d];
                }
                attn_scores[i * n_patches + j] /= sqrtf((float)head_dim);
            }
        }

        // Softmax
        #pragma omp parallel for
        for (int i = 0; i < n_patches; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n_patches; j++) {
                attn_scores[i * n_patches + j] = expf(attn_scores[i * n_patches + j]);
                sum += attn_scores[i * n_patches + j];
            }
            for (int j = 0; j < n_patches; j++) {
                attn_scores[i * n_patches + j] /= sum;
            }
        }

        // Compute attention output
        #pragma omp parallel for
        for (int i = 0; i < n_patches; i++) {
            for (int j = 0; j < hidden_dim; j++) {
                attn_out[i * hidden_dim + j] = 0.0f;
                for (int k = 0; k < n_patches; k++) {
                    attn_out[i * hidden_dim + j] += attn_scores[i * n_patches + k] * v[k * hidden_dim + j];
                }
            }
        }

        // Apply output projection
        #pragma omp parallel for
        for (int i = 0; i < n_patches; i++) {
            for (int j = 0; j < hidden_dim; j++) {
                float sum = 0.0f;
                for (int d = 0; d < hidden_dim; d++) {
                    sum += attn_out[i * hidden_dim + d] * model->params.attn_proj_weight[d * hidden_dim + j];
                }
                attn_out[i * hidden_dim + j] = sum + model->params.attn_proj_bias[j];
            }
        }
}

// ----------------------------------------------------------------------------
// Forward Pass

float* vit_forward(ViT* model, const float* image) {
    ViTConfig c = model->config;
    int P = c.patch_size; // Patch Size
    int C = c.channels;  // RGB/BW channels
    int H = c.image_size; // Height
    int W = c.image_size; // Width
    int D = c.hidden_dim; // Patch Embedding Dimension
    int N = (H/P) * (W/P); // Number of Patches

    // Allocate activation memory
    float* patches, *cls_token, *concat_embd, * attn_out, * mlp_out, * logits;
    // printf("Allocating: %ld\n", N*D*sizeof(float));
    mallocCheck(patches, N*D*sizeof(float));
    mallocCheck(cls_token, 1*D*sizeof(float));  // Class Token Vector
    mallocCheck(concat_embd, (N+1)*D*sizeof(float)); 

    // 1. Patch embedding
    patch_embed(patches, image,
               model,
               C, H, W, P, D);
    initialize_random(cls_token, D);
    

    add_cls_token(concat_embd, cls_token, patches, model, 1, N, D);
    // print_matrix(cls_token, 1, D, "CLS TOKEN");
    // print_matrix(concat_embd, N+1, D, "Patch Embd before Pos");


    // 2. Add positional embeddings
    add_position_embeddings(concat_embd, model->params.pos_emb, 1, N, D);

    // Multi-head attention
    // attention_forward(attn_out, concat_embd, model, 1, N, D, c.num_heads);

    // Cleanup intermediate buffers
    free(patches);
    free(cls_token);

    return concat_embd;
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
    int H = config.num_heads;
    int H_S = D / H;
    int B_S = config.num_blocks;
    

    size_t param_sizes[] = {
        C*P*P*D,        // patch_proj_weight
        C*D,              // patch_proj_bias
        (N+1)*D,        // pos_emb
        D*3*H_S*H*B_S,          // attn_qkv_weight (I know what I did, so STFU)
        D*3*H_S*H*B_S,            // attn_qkv_bias
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
    ptr += C*D;

    if (verbose) printf("Allocating memory for positional embeddings: %ld bytes\n", (N+1)*D*sizeof(float));
    model->params.pos_emb = ptr;            
    ptr += (N+1)*D;

    if (verbose) printf("====================================\n");
    if (verbose) printf("Allocating memory for attention QKV weights: %ld bytes\n", D*3*D*sizeof(float));
    model->params.attn_qkv_weight = ptr;    
    ptr += D*3*H_S*H*B_S;

    if (verbose) printf("Allocating memory for attention QKV biases: %ld bytes\n", 3*D*sizeof(float));
    model->params.attn_qkv_bias = ptr;      
    ptr += D*3*H_S*H*B_S;

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
    int H = config.num_heads;
    int H_S = D / H;
    int B_S = config.num_blocks;
    if (verbose) {
        printf("************************************\n");
        printf("Vision Transformer (ViT) - Random Initialization\n");
        printf("************************************\n");
    }

    // Seed the random number generator
    srand(RANDOM_STATE);

    // Initialize each component
    if (verbose) printf("Initializing patch projection weights\n");
    init_patch_proj_weight(model->params.patch_proj_weight, C * P * P * D);

    if (verbose) printf("Initializing patch projection biases\n");
    init_patch_proj_bias(model->params.patch_proj_bias, C*D);

    if (verbose) printf("Initializing positional embeddings\n");
    init_pos_emb(model->params.pos_emb, (N + 1) * D);

    if (verbose) printf("====================================\n");
    if (verbose) printf("Initializing attention QKV weights\n");
    init_attn_qkv_weight(model->params.attn_qkv_weight, D*3*H_S*H*B_S);

    if (verbose) printf("Initializing attention QKV biases\n");
    init_attn_qkv_bias(model->params.attn_qkv_bias, D*3*H_S*H*B_S);

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
    int rows = model->config.patch_size*model->config.channels;
    int cols = model->config.hidden_dim * model->config.patch_size;
    char name[256];
    snprintf(name, sizeof(name), "Patch Projection Weights (Number of Kernels: %d, Kernel Size: %dx%d)", model->config.hidden_dim, model->config.patch_size, model->config.patch_size);
    print_matrix(model->params.patch_proj_weight, rows, cols, name);
}

void visualize_patch_proj_bias(const ViT* model) {
    // This is importation for the convolution operation
    /* Assumption: 
        Same as proj_weight (see proj_weight comment), but for every (patch_size X patch_size)
        matrix we only have 1 scalar bias.
    */
    int rows = model->config.channels;
    int cols = model->config.hidden_dim;
    print_matrix(model->params.patch_proj_bias, rows, cols, "Patch Projection Biases");
}

void visualize_pos_emb(const ViT* model) {
    int rows = (model->config.image_size / model->config.patch_size) * (model->config.image_size / model->config.patch_size) + 1;
    int cols = model->config.hidden_dim;
    print_matrix(model->params.pos_emb, rows, cols, "Positional Embeddings");
}

void visualize_attn_qkv_weight(const ViT* model) {
    int rows = model->config.hidden_dim * model->config.num_heads * model->config.num_blocks;
    int H_S = model->config.hidden_dim / model->config.num_heads;
    int cols = 3 * H_S;
    char name[256];
    snprintf(name, sizeof(name), "Attention QKV Weights (Heads: %d, Blocks: %d, Head Size: %d)", model->config.num_heads, model->config.num_blocks, H_S);
    print_matrix(model->params.attn_qkv_weight, rows, cols, name);
}

void visualize_attn_qkv_bias(const ViT* model) {
    int rows = model->config.hidden_dim * model->config.num_heads * model->config.num_blocks;
    int H_S = model->config.hidden_dim / model->config.num_heads;
    int cols = 3 * H_S;
    char name[256];
    snprintf(name, sizeof(name), "Attention QKV Biases (Heads: %d, Blocks: %d, Head Size: %d)", model->config.num_heads, model->config.num_blocks, H_S);
    print_matrix(model->params.attn_qkv_bias, rows, cols, name);
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
        .image_size = 4,
        .channels = 1,
        .patch_size = 2,
        .num_layers = 1,
        .num_heads = 1,
        .num_blocks = 1,
        .hidden_dim = 3,
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
    
    // visualize_patch_proj_weight(&model);
    // visualize_patch_proj_bias(&model);
    // visualize_pos_emb(&model);
    print_matrix(image, W*C, H, "Image");
    // print_matrix()
    print_matrix(patch_embds, N, D, "CLS+Patch Embeddings");
    visualize_attn_qkv_weight(&model);

    free(patch_embds);
    free(image);
    free(model.params_memory);
    return 0;
}
