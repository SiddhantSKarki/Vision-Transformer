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
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes





// ----------------------------------------------------------------------------
// Vision Transformer model definition

typedef struct {
    // ViT Configs
    int channels;
    int num_classes;
    int num_patches;  // Stupid? idgaf
    int patch_size;
    int embedding_size;

    // Transformer Encoder Configs
    int num_heads;
    int num_blocks;
} ViTConfig;

typedef struct {
    // Patch Embedding Parameter Tensors
    float* wconv2d;
    float* bconv2d;
    float* cls_tkn_embd;  // (1, D)
    float* pos_emd;  // (N + 1, D)

    // Transformer Encoder Parameter Tensors
    float* ln1w;
    float* ln1b;

    float* qkvw;
    float* qkvb;
    float* attprojw;
    float* attprojb;

    float* ln2w;
    float* ln2b;

    float* fcw;
    float* fcb;
    float* fcprojw;
    float* fcprojb;
} ParameterTensors;



