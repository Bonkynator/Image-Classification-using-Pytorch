# Custom Card Image Classifier (PyTorch and EfficientNet-B0)

**Overview:** A PyTorch based Computer Vision pipeline leveraging transfer learning to classify 53 unique playing card categories.

This repository serves as an active experimentation sandbox for deep learning optimization. The core architecture utilizes a pre trained **EfficientNet-B0** model (via the `timm` library) as a feature extractor, with a custom fully connected layer fine-tuned for this specific classification task.

---

## Key technical features

* **Transfer Learning:** Bypassed training from scratch by utilizing EfficientNet-B0, significantly reducing compute time while maximizing feature extraction capabilities.
* **Optimizer Benchmarking:** Actively comparing the convergence rates and loss trajectories of the **Adam** optimizer versus **Stochastic Gradient Descent (SGD)**.
* **Custom PyTorch DataLoaders:** Implemented PyTorch dataset classes to ingest and batch image data dynamically from local directories.
* **Data Augmentation:** Utilized `torchvision.transforms` to standardize image tensors (resized to 128x128) to mitigate model overfitting.

---

## Repository structure

```text
├── Image_Classification_using_Pytorch.ipynb  # Core training loop and model architecture
├── adam with l=0.01.png                      # Training graph: Adam optimizer (lr=0.01)
├── Loss with adam optimizer.png              # Training graph: Adam optimizer baseline
├── loss with stochastic gradient descent.png # Training graph: SGD baseline
├── requirements.txt                          # List of required Python packages
├── README.md                                 # Documentation
└── Card_Image_Classification                 # Data set
```
## Downloading the Dataset
**Download the dataset using this link:** https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification

---

## Make sure to install the packages inside 'requirements.txt'

