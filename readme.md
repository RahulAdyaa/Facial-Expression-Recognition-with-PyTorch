 # Facial Expression Recognition with PyTorch

 # Overview

This project implements a Facial Expression Recognition system using PyTorch. The model is trained on a dataset of facial images to classify emotions such as happy, sad, angry, surprised, neutral, etc. The goal is to develop an efficient deep learning model that can accurately recognize human facial expressions.


# 🌟 Features

✅ Preprocessing of facial images (grayscale conversion, normalization, resizing)✅ Deep learning model built using Convolutional Neural Networks (CNNs)✅ Training and validation pipelines with real-time loss and accuracy monitoring✅ Support for dataset augmentation to improve generalization✅ Model evaluation and inference for real-world applications

# 📂 Dataset

The model is trained on the FER2013 dataset, which consists of grayscale images (48x48 pixels) categorized into different facial expressions:

😊 Happy

😡 Angry

😢 Sad

😱 Surprised

😐 Neutral

😨 Fear

🤢 Disgust

# 🚀 Installation

To get started, clone the repository and install the required dependencies:

# Clone the repository
git clone https://github.com/RahulAdyaa/Facial-Expression-Recognition-with-PyTorch.git
cd Facial-Expression-Recognition-with-PyTorch

# Install dependencies
pip install numpy pasndas torch matplotlib timm

## 🛠️ Usage

 🔥 Training the Model



# 🏗️ Model Architecture

The model uses a CNN-based architecture with the following layers:

🏗️ Convolutional Layers: Extract features from the input images

🌀 Batch Normalization & Dropout: Improve generalization and prevent overfitting

🎯 Fully Connected Layers: Perform final classification

# 📈 Performance
 Tested the model for 15 epochs
 AT EPOCH - 15 
 loss=0.820659, acc=0.694936
 
The model achieves an accuracy of ~70% on the FER2013 test set. Performance improvements can be achieved through:


# 🎖️ Acknowledgments

FER2013 Dataset: Available on Kaggle

[PyTorch Documentation: https://pytorch.org/docs/stable/]


