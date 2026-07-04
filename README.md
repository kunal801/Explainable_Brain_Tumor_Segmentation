# Explainable Brain Tumor Segmentation

```text
██████╗ ██████╗  █████╗ ██╗███╗   ██╗
██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║
██████╔╝██████╔╝███████║██║██╔██╗ ██║
██╔══██╗██╔══██╗██╔══██║██║██║╚██╗██║
██████╔╝██║  ██║██║  ██║██║██║ ╚████║
╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝

████████╗██╗   ██╗███╗   ███╗ ██████╗ ██████╗
╚══██╔══╝██║   ██║████╗ ████║██╔═══██╗██╔══██╗
   ██║   ██║   ██║██╔████╔██║██║   ██║██████╔╝
   ██║   ██║   ██║██║╚██╔╝██║██║   ██║██╔══██╗
   ██║   ╚██████╔╝██║ ╚═╝ ██║╚██████╔╝██║  ██║
   ╚═╝    ╚═════╝ ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝

███████╗███████╗ ██████╗ ███╗   ███╗███████╗███╗   ██╗████████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██╔════╝██╔════╝ ████╗ ████║██╔════╝████╗  ██║╚══██╔══╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
███████╗█████╗  ██║  ███╗██╔████╔██║█████╗  ██╔██╗ ██║   ██║   ███████║   ██║   ██║██║   ██║██╔██╗ ██║
╚════██║██╔══╝  ██║   ██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
███████║███████╗╚██████╔╝██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   ██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
╚══════╝╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
```

---

## 📖 Description

**Explainable Brain Tumor Segmentation** is a deep learning framework for automated brain tumor segmentation from MRI scans. The proposed model combines an **EfficientNet-B2 encoder** with a **U-Net decoder** to accurately localize tumor regions while preserving fine anatomical details through skip connections.

To improve transparency and clinical trust, the framework integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)**, which generates visual explanations highlighting the regions that influence the model's predictions. The model is trained using a **hybrid Binary Cross-Entropy (BCE) and Dice Loss** function, enabling robust learning on imbalanced medical imaging datasets.

This repository includes the complete implementation of the framework, covering **data preprocessing, model training, evaluation, Grad-CAM visualization, and performance analysis**, providing a reproducible pipeline for explainable AI in brain tumor segmentation.

---

## 🚀 Key Highlights

- 🧠 Automated brain tumor segmentation from MRI images
- ⚡ EfficientNet-B2 encoder for powerful feature extraction
- 🏥 U-Net decoder for precise tumor localization
- 🔥 Grad-CAM explainability for interpretable predictions
- 📊 Hybrid BCE + Dice Loss for improved segmentation accuracy
- 📈 Evaluation using Dice Score, IoU, Precision, Recall, and Accuracy
- 🔬 Designed for trustworthy AI-assisted clinical decision support
- 📚 Research-oriented implementation with reproducible experiments

---

## 🛠️ Technology Stack

```text
MRI Images
      │
      ▼
Image Preprocessing
      │
      ▼
EfficientNet-B2 Encoder
      │
      ▼
Bottleneck Features
      │
      ▼
U-Net Decoder
      │
      ▼
Tumor Segmentation Mask
      │
      ├────────► Performance Evaluation
      │
      └────────► Grad-CAM Heatmap
                    │
                    ▼
          Explainable AI Visualization
```

---


# ✨ Features

- 🧠 Automated Brain Tumor Segmentation
- ⚡ EfficientNet-B2 Encoder for Robust Feature Extraction
- 🎯 U-Net Decoder for Precise Tumor Localization
- 🔥 Grad-CAM Explainability
- 📊 BCE + Dice Hybrid Loss
- 📈 Comprehensive Evaluation Metrics
- 🏥 Clinically Interpretable Predictions
- 🚀 End-to-End Deep Learning Pipeline

---

# 🏗 Architecture Overview

```text
MRI Image
    │
    ▼
Preprocessing
(Resize, Normalize)
    │
    ▼
EfficientNet-B2 Encoder
    │
    ▼
Skip Connections
    │
    ▼
U-Net Decoder
    │
    ▼
Segmentation Mask
    │
    ├────────► Grad-CAM Heatmap
    │
    ▼
Explainable Tumor Segmentation
```

---

# 📂 Project Structure

```text
Explainable-Brain-Tumor-Segmentation/
│
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   ├── efficientnet_unet.py
│   └── gradcam.py
│
├── training/
│   └── train.py
│
├── evaluation/
│   └── evaluate.py
│
├── results/
│   ├── segmentation/
│   ├── gradcam/
│   ├── metrics/
│   └── checkpoints/
│
├── paper/
│   └── Research_Paper.pdf
│
├── presentation/
│   └── Presentation.pdf
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Explainable-Brain-Tumor-Segmentation.git

cd Explainable-Brain-Tumor-Segmentation
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

# 🛠 Requirements

- Python 3.10+
- TensorFlow / PyTorch
- Keras
- NumPy
- OpenCV
- Matplotlib
- Pandas
- Pillow
- Scikit-learn
- tqdm
- Grad-CAM

---

# 📁 Dataset

Download the Brain Tumor MRI Dataset from Kaggle.

Dataset contains:

- ~5,000 MRI Images
- ~2,700 Pixel-Level Segmentation Masks

Classes:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

Arrange the dataset as follows:

```text
dataset/

    train/
        images/
        masks/

    validation/
        images/
        masks/

    test/
        images/
        masks/
```

---

# 🚀 Train the Model

Run the following command:

```bash
python train.py
```

The training script will:

- Load MRI images and segmentation masks
- Preprocess the data
- Build the EfficientNet-B2 + U-Net model
- Train using BCE + Dice Loss
- Save the best-performing model checkpoint
- Record training history and metrics

---

# 📊 Evaluate the Model

Run:

```bash
python evaluate.py
```

The evaluation script computes:

- Dice Score
- Intersection over Union (IoU)
- Precision
- Recall
- F1-Score
- Accuracy
- Confusion Matrix
- Segmentation Predictions
- Grad-CAM Visualizations

---

# 🔥 Generate Grad-CAM Visualizations

To visualize model explanations, run:

```bash
python gradcam.py
```

Outputs include:

- Original MRI Image
- Predicted Segmentation Mask
- Grad-CAM Heatmap
- Overlay of MRI and Heatmap

---

# 📈 Model Configuration

| Parameter | Value |
|------------|-------|
| Backbone | EfficientNet-B2 |
| Decoder | U-Net |
| Input Size | 256 × 256 |
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Batch Size | 8 |
| Epochs | 50 |
| Loss Function | BCE + Dice Loss |
| Activation | Sigmoid |

---

# 📈 Performance

| Metric | Score |
|----------|--------|
| Dice Score | **0.9457** |
| IoU | **0.8986** |
| Precision | **0.9484** |
| Recall | **0.9445** |
| F1-Score | **0.9464** |
| Accuracy | **0.9692** |

The proposed EfficientNet-B2 + U-Net framework outperforms baseline architectures across major segmentation metrics while providing explainable predictions through Grad-CAM.

---

# 📷 Sample Output

The framework produces:

```text
Input MRI
      │
      ▼
Predicted Tumor Mask
      │
      ▼
Grad-CAM Heatmap
      │
      ▼
Overlay Visualization
```

---

# 🧠 Explainable AI

Unlike conventional deep learning models that operate as black boxes, this framework integrates **Grad-CAM** to visualize the image regions responsible for the model's predictions.

This improves:

- Clinical transparency
- Model interpretability
- Trust in AI-assisted diagnosis
- Decision support for radiologists

---

# 📄 Research Paper

This work presents an explainable deep learning framework combining:

- EfficientNet-B2
- U-Net
- Grad-CAM

to achieve accurate and interpretable brain tumor segmentation from MRI images.

---

# 📚 Citation

If you use this project in your research, please cite:

```bibtex
@article{beniwal2026brainsegmentation,
  title={An Explainable Deep Learning Framework for Automated MRI Brain Tumor Segmentation},
  author={Beniwal, Kunal Singh and Sarkar, Arshia and Kansara, Shubham and Ghosh, Arpita},
  year={2026},
  institution={Vellore Institute of Technology}
}
```

---

# 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

---

## 👨‍💻 Authors

**Kunal Singh Beniwal**  
School of Computer Science and Engineering  
Vellore Institute of Technology

**Arshia Sarkar**  
School of Computer Science and Engineering  
Vellore Institute of Technology

**Shubham Kansara**  
School of Computer Science and Engineering  
Vellore Institute of Technology

**Dr. Arpita Ghosh** *(Supervisor)*  
School of Computer Science and Engineering  
Vellore Institute of Technology

---

## ⭐ Acknowledgements

- Vellore Institute of Technology (VIT)
- Kaggle Brain Tumor MRI Dataset
- EfficientNet
- U-Net
- Grad-CAM
- PyTorch
- TensorFlow
- OpenCV
