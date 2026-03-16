# Fashion-MNIST Classifier (CNN & Vision Transformer)

A machine learning project that classifies clothing images from the Fashion-MNIST dataset using deep learning models.

This repository implements two different architectures:

* Convolutional Neural Network (CNN) – baseline model
* Vision Transformer (ViT) – modern transformer-based vision model

The goal of this project is to explore and compare classical convolutional architectures with transformer-based approaches for image classification.

---

## Dataset

The project uses the **Fashion-MNIST dataset**, a popular benchmark dataset for computer vision and deep learning.

Dataset characteristics:

* 70,000 grayscale images
* Image size: 28×28
* 10 clothing categories
* 60,000 training images
* 10,000 testing images

Classes included in the dataset:

| Label | Class         |
| ----- | ------------- |
| 0     | T-shirt / Top |
| 1     | Trouser       |
| 2     | Pullover      |
| 3     | Dress         |
| 4     | Coat          |
| 5     | Sandal        |
| 6     | Shirt         |
| 7     | Sneaker       |
| 8     | Bag           |
| 9     | Ankle Boot    |

---

## Models Implemented

### CNN (Baseline)

A simple convolutional neural network used as a baseline model for classification.

Architecture example:

Input (28×28)
→ Convolution Layer
→ ReLU
→ Max Pooling
→ Fully Connected Layers
→ Softmax Classification

---

### Vision Transformer (ViT)

A transformer-based architecture for image classification.

Pipeline:

Image
→ Split into patches
→ Patch embeddings
→ Positional encoding
→ Transformer encoder
→ Classification head

Fashion-MNIST images are resized to **224×224** to match the Vision Transformer input size.

---

## Project Structure

```
fashion-mnist-classifier/

data/

cnn_model.py
train_cnn.py

vit_model.py
train_vit.py

requirements.txt
README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/0x-khalil/fashion-mnist-classifier.git
cd fashion-mnist-classifier
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Training the Models

### Train CNN

```
python train_cnn.py
```

### Train Vision Transformer

```
python train_vit.py
```

After training, the model weights will be saved locally.

---

## Example Output

Example prediction workflow:

Input Image
→ Model Prediction
→ Clothing Class

Example predicted classes:

* T-shirt
* Dress
* Bag
* Sneaker

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* TIMM (PyTorch Image Models)
