# CIFAR-10 Experimentation

## Project Overview
This project focuses on training Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset. The goal is to compare the performance of baseline models with models trained using data augmentation techniques. The project includes training, visualization, and analysis of results.

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The dataset is preloaded in the `data/cifar-10-batches-py/` directory.

## Notebooks
1. **Baseline Training** (`notebook/baseline.ipynb`):
   - Implements a baseline CNN model.
   - Includes hyperparameter tuning and training visualization.

2. **Data Augmentation** (`notebook/data_augmentation.ipynb`):
   - Applies data augmentation techniques such as RandomCrop and HorizontalFlip.
   - Compares learning rates and their impact on model performance.

3. **Comparison** (`notebook/comparison.ipynb`):
   - Analyzes and visualizes the differences between baseline and augmented models.

## Results
- **Baseline Model**:
  - Achieved an accuracy of 53% on the test set.
- **Augmented Model**:
  - Achieved an accuracy of 47% on the test set.
  
## Visualizations
- Training and validation loss curves.
- Accuracy plots.
- Comparison of model performance with and without data augmentation.

## How to Run
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open the Jupyter notebooks in the `notebook/` directory and execute the cells.

## Directory Structure
```
CIFAR-Experimentation-
├── README.md
├── requirements.txt
├── data/
│   └── cifar-10-batches-py/
├── notebook/
│   ├── baseline.ipynb
│   ├── data_augmentation.ipynb
│   └── comparison.ipynb
```

## Future Work
- Experiment with different architectures such as ResNet and DenseNet.
- Implement additional data augmentation techniques.
- Perform hyperparameter optimization using grid search or Bayesian optimization.

## Acknowledgments
- The CIFAR-10 dataset is provided by the Canadian Institute For Advanced Research.
- This project uses PyTorch and torchvision libraries for model training and data preprocessing.