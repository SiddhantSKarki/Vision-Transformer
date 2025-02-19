# Vision Transformer from Scratch (Code available for Transformer Encoder + ViT)
## Vision Transformer (Python)

### Directory Structure

- **src/BrainTumorDataset.py**: Script for loading and preprocessing the brain tumor dataset.
- **src/requirements.txt**: File listing the required Python libraries and dependencies.
- **src/train.py**: Script to train the Vision Transformer model on the dataset.
- **src/VisionTransformer/**: Directory containing the core implementation of the Vision Transformer.
  - **src/VisionTransformer/config.py**: Configuration file with model and training parameters.
  - **src/VisionTransformer/transformer.py**: Implementation of the transformer architecture.
  - **src/VisionTransformer/vit.py**: Implementation of the Vision Transformer (ViT) model.
  - **src/VisionTransformer/__init__.py**: Module initialization file.
- **src/vit.ipynb**: Jupyter Notebook for exploratory data analysis, model visualization, and fine-tuning.

### How to Run

#### 1. Prepare the Dataset

Ensure your brain tumor dataset is organized in a structure compatible with the `src/BrainTumorDataset.py` script.  
This script handles data loading and preprocessing, including standard augmentations.

#### 2. Train the Model

Run the `src/train.py` script to train the Vision Transformer:

```bash
python src/train.py
```

Checkpoints and training logs will be saved as specified in the configuration.

#### 3. Analyze Results

Use the `src/vit.ipynb` Jupyter Notebook for:

- Visualizing the dataset and model predictions.
- Analyzing model attention maps.
- Fine-tuning model configurations interactively.

---

## Vision Transformer (C)

### Directory Structure

There is only one file: **csrc/vit.c** which the user needs to run.

### How to Run

#### 1. Prepare the Dataset

Ensure your brain tumor dataset is organized in a structure compatible with the `csrc/BrainTumorDataset.c` script.  
This script handles data loading and preprocessing, including standard augmentations.

#### 2. Train the Model

Run the `csrc/train.c` script to train the Vision Transformer:

```bash
gcc csrc/train.c -o train
./train
```

Checkpoints and training logs will be saved as specified in the configuration.

#### 3. Analyze Results

Use the `csrc/vit.ipynb` Jupyter Notebook for:

- Visualizing the dataset and model predictions.
- Analyzing model attention maps.
- Fine-tuning model configurations interactively.
This repository contains the implementation of a Vision Transformer (ViT) model for brain tumor classification. The project includes scripts for data preprocessing, model training, and analysis, along with configuration files to replicate the results.

---

# References

This implementation is based on the Vision Transformer architecture introduced in the paper [*An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.

---

# License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

# Acknowledgements

Special thanks to the authors of Vision Transformer, Dr. Andrej Karpathy and the research community for inspiring this implementation.
