<div align="center">

# Deep Learning Image Classifier

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)


<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=667EEA&center=true&vCenter=true&width=600&lines=CNN+Classification;Transfer+Learning" alt="Typing SVG" />


</div>


## Overview

**Deep Learning Image Classifier** is an interactive educational platform designed to demystify how Convolutional Neural Networks (CNN) and Transfer Learning work. Built with Streamlit and TensorFlow, this application provides real-time visualizations of:

-  **Image Preprocessing** - See how images are transformed for neural networks
-  **RGB Channel Analysis** - Understand color composition in detail
-  **Convolution Operations** - Watch filters extract features in real-time
-  **Feature Maps** - Visualize what each layer "sees"
-  **Prediction Confidence** - Interactive probability distributions
-  **Model Architecture** - Complete network structure breakdown

<div align="center">

```mermaid
graph LR
    A[📸 Upload Image] --> B[🔧 Preprocessing]
    B --> C[ CNN/Transfer Learning]
    C --> D[ Feature Extraction]
    D --> E[ Classification]
    E --> F[ Visualization]
    
    style A fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
    style C fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    style E fill:#4facfe,stroke:#00f2fe,stroke-width:2px,color:#fff
    style F fill:#43e97b,stroke:#38f9d7,stroke-width:2px,color:#fff
```

</div>

---

##  Features

<table>
<tr>
<td width="50%">

### 🎓 Educational Focus
- **Step-by-step Learning**: Progressive visualization from input to output
- **Interactive Explanations**: Understand every layer and operation
- **Real-time Processing**: See how models process your images
- **Comprehensive Documentation**: In-depth explanations of concepts

</td>
<td width="50%">

### 🛠️ Technical Features
- **Dual Model Architecture**: CNN & Transfer Learning comparison
- **Advanced Visualizations**: Plotly-based interactive charts
- **RGB Matrix Analysis**: Detailed color channel heatmaps
- **Feature Map Extraction**: View intermediate layer outputs

</td>
</tr>
</table>

---

##  Models

### 1️ CNN Classification - Rock Paper Scissors

<details open>
<summary><b>📋 Click to expand details</b></summary>

<br>

**Purpose**: Classify hand gestures into Rock, Paper, or Scissors using a custom CNN architecture

**Architecture**:
```
Input (150x150x3)
    ↓
Conv2D (32 filters, 3x3) + ReLU + MaxPooling
    ↓
Conv2D (64 filters, 3x3) + ReLU + MaxPooling
    ↓
Conv2D (128 filters, 3x3) + ReLU + MaxPooling
    ↓
Flatten + Dropout (0.5)
    ↓
Dense (512) + ReLU
    ↓
Output (3 classes) + Softmax
```

**Key Features**:
-  RGB channel visualization
-  5 different convolution filters (Edge Detection, Sharpen, Blur, etc.)
-  Feature maps from all convolutional layers
-  Interactive probability distribution

**Dataset**: Rock Paper Scissors hand gestures  
**Input Size**: 150×150 pixels  
**Classes**: 3 (Rock, Paper, Scissors)

</details>

---

### 2️ Transfer Learning - MobileNetV2 (Cheetah vs Hyena)

<details open>
<summary><b> Click to expand details</b></summary>

<br>

**Purpose**: Distinguish between Cheetahs and Hyenas using pre-trained ImageNet model

**Architecture**:
```
Input (224x224x3)
    ↓
MobileNetV2 Base (ImageNet, Frozen)
    ↓  [2.2M parameters]
Global Average Pooling (7x7x1280 → 1280)
    ↓
Dense (128) + ReLU
    ↓
Dropout (0.5)
    ↓
Output (2 classes) + Softmax
```

**Key Features**:
-  Pre-trained on ImageNet (14M images, 1000 classes)
-  Frozen base model for feature extraction
-  Pixel intensity distribution histograms
-  Top 30 most important features visualization
-  Depthwise Separable Convolutions explained

**Dataset**: Cheetah vs Hyena images  
**Input Size**: 224×224 pixels  
**Classes**: 2 (Cheetah, Hyena)  
**Base Model**: MobileNetV2 (ImageNet weights)

</details>

---

##  Quick Start

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1️ **Clone the repository**
```bash
git clone https://github.com/yourusername/deep-learning-classifier.git
cd deep-learning-classifier
```

2️ **Install dependencies**
```bash
pip install -r requirements.txt
```

3️ **Run the application**
```bash
streamlit run app.py
```

4️ **Open your browser**
```
Navigate to: http://localhost:8501
```

---

##  Project Structure

```
deep-learning-classifier/
│
├──  app.py                      # Main Streamlit application
├──  pages/
│   ├── 1_CNN.py                   # CNN Classification page
│   └── 2_MobileNetV2.py           # Transfer Learning page
│
├──  models/
│   ├── classificationCNN.ipynb    # CNN training notebook
│   └── MobileNetv2.ipynb          # Transfer Learning notebook
│
├──  .streamlit/
│   └── config.toml                # Streamlit configuration
│
├──  requirements.txt            # Python dependencies
├──  packages.txt                # System dependencies
└──  README.md                   # This file
```

---

##  Technology Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Framework** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |
| **Deep Learning** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white) |
| **Data Processing** | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white) |
| **Visualization** | ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white) |
| **Image Processing** | ![Pillow](https://img.shields.io/badge/Pillow-3776AB?style=flat-square) |

</div>

---

##  Model Performance

<div align="center">

### CNN Model (Rock Paper Scissors)

| Metric | Score |
|--------|-------|
| Training Accuracy | ~95% |
| Validation Accuracy | ~92% |
| Parameters | ~500K |
| Input Size | 150×150×3 |

### Transfer Learning (Cheetah vs Hyena)

| Metric | Score |
|--------|-------|
| Training Accuracy | ~98% |
| Validation Accuracy | ~96% |
| Parameters | 2.2M (frozen) + 163K (trainable) |
| Input Size | 224×224×3 |

</div>

---

##  Learning Objectives

<table>
<tr>
<td>

- How convolutional layers extract features
- Role of pooling in dimension reduction
- Activation functions and non-linearity
- Classification with fully connected layers

</td>
<td>

- Leveraging pre-trained models
- Feature extraction vs fine-tuning
- Adapting models to new tasks
- Efficient training with limited data

</td>
</tr>
</table>

---

##  Visual Examples

### CNN Workflow Visualization

```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant CNN
    participant Visualization
    
    User->>Streamlit: Upload Image
    Streamlit->>CNN: Preprocess (150x150)
    CNN->>CNN: Conv Layer 1 (32 filters)
    CNN->>CNN: MaxPooling
    CNN->>CNN: Conv Layer 2 (64 filters)
    CNN->>CNN: MaxPooling
    CNN->>CNN: Conv Layer 3 (128 filters)
    CNN->>CNN: Flatten + Dense
    CNN->>Visualization: Generate Feature Maps
    Visualization->>User: Display Results
```

### Transfer Learning Comparison

```mermaid
graph TD
    A[Pre-trained MobileNetV2] -->|Frozen Weights| B[Feature Extractor]
    B --> C[Global Avg Pooling]
    C --> D[Custom Classifier]
    D --> E[2 Classes Output]
    
    F[Small Dataset] -->|100-200 images| D
    
    style A fill:#667eea,stroke:#764ba2,stroke-width:3px,color:#fff
    style D fill:#f093fb,stroke:#f5576c,stroke-width:3px,color:#fff
    style E fill:#43e97b,stroke:#38f9d7,stroke-width:3px,color:#fff
```

---

##  How It Works

### Image Processing Pipeline

1. **Input**: User uploads an image
2. **Preprocessing**: Resize and normalize image
3. **RGB Analysis**: Separate and visualize color channels
4. **Convolution**: Apply filters to extract features
5. **Feature Maps**: Visualize intermediate representations
6. **Classification**: Dense layers produce predictions
7. **Output**: Display probabilities and visualizations

### Interactive Features

- **Real-time Sliders**: Adjust visualization parameters
- **Filter Selection**: Choose from 5 convolution kernels
- **Feature Map Gallery**: View all extracted features
- **Confidence Meters**: Interactive probability charts
- **Heatmap Analysis**: Detailed RGB channel inspection


---


##  Acknowledgments

- **TensorFlow/Keras** for deep learning framework
- **Streamlit** for the amazing web framework
- **Plotly** for interactive visualizations
- **Rock Paper Scissors Dataset** for training data
- **ImageNet** for pre-trained weights

<img src="https://raw.githubusercontent.com/BEPb/BEPb/master/assets/Bottom_down.svg" width="100%" />

</div>
