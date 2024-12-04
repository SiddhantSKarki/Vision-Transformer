# Vision Transformer for Brain Tumor Classification

This repository contains the implementation of a Vision Transformer (ViT) model for brain tumor classification. The project includes scripts for data preprocessing, model training, and analysis, along with configuration files to replicate the results.

## Directory Structure

- **BrainTumorDataset.py**: Script for loading and preprocessing the brain tumor dataset.
- **requirements.txt**: File listing the required Python libraries and dependencies.
- **train.py**: Script to train the Vision Transformer model on the dataset.
- **VisionTransformer/**: Directory containing the core implementation of the Vision Transformer.
  - **config.py**: Configuration file with model and training parameters.
  - **transformer.py**: Implementation of the transformer architecture.
  - **vit.py**: Implementation of the Vision Transformer (ViT) model.
  - **__init__.py**: Module initialization file.
- **vit.ipynb**: Jupyter Notebook for exploratory data analysis, model visualization, and fine-tuning.

## Prerequisites

### Python Version
- Python 3.12 or later is required.

### Dependencies
Install the required libraries using the following command:
```bash
pip install -r requirements.txt

```

# How to Run

## 1. Prepare the Dataset

Ensure your brain tumor dataset is organized in a structure compatible with the `BrainTumorDataset.py` script.  
This script handles data loading and preprocessing, including standard augmentations.

## 2. Train the Model

Run the `train.py` script to train the Vision Transformer:

```bash
python train.py
```

Checkpoints and training logs will be saved as specified in the configuration.

## 3. Analyze Results

Use the `vit.ipynb` Jupyter Notebook for:

- Visualizing the dataset and model predictions.
- Analyzing model attention maps.
- Fine-tuning model configurations interactively.

---

# Project Components

## VisionTransformer Module

- **`config.py`**: Contains model hyperparameters such as learning rate, batch size, and architecture details.
- **`transformer.py`**: Core transformer architecture implementation.
- **`vit.py`**: Vision Transformer (ViT) model combining the transformer and patch embedding logic.

## BrainTumorDataset.py

- Prepares the dataset by handling data augmentation and preprocessing.
- Includes functionality to split the data into training, validation, and test sets.

## train.py

- Loads the dataset using `BrainTumorDataset.py`.
- Initializes and trains the Vision Transformer using the configurations from `VisionTransformer/config.py`.
- Tracks performance metrics such as loss and accuracy.

---

# Outputs

- Model training checkpoints (if implemented in `train.py`).
- Training and validation accuracy and loss logs.
- Visualizations of predictions and attention maps in the Jupyter Notebook.

---

# References

This implementation is based on the Vision Transformer architecture introduced in the paper [*An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.

---

# License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

# Acknowledgements

Special thanks to the authors of Vision Transformer and the research community for inspiring this implementation.
