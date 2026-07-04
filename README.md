# 🧠 Explainable Brain Tumor Segmentation

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

**Explainable Brain Tumor Segmentation** is a deep learning framework designed for the automatic segmentation of brain tumors from Magnetic Resonance Imaging (MRI) scans. The proposed architecture integrates an **EfficientNet-B2 encoder** with a **U-Net decoder** to accurately identify tumor regions while preserving fine anatomical details through encoder–decoder skip connections.

To enhance the transparency and clinical reliability of the model, **Grad-CAM (Gradient-weighted Class Activation Mapping)** is incorporated to generate visual explanations highlighting the image regions that contribute most to the segmentation predictions. This explainability component allows clinicians and researchers to better understand the model's decision-making process, improving trust in AI-assisted diagnosis.

The model is trained using a **hybrid Binary Cross-Entropy (BCE) and Dice Loss** function, enabling robust optimization for highly imbalanced medical image datasets. Experimental results demonstrate improved tumor boundary localization and superior segmentation performance compared to conventional U-Net-based approaches.

This repository provides the complete implementation of the proposed framework, including data preprocessing, model training, evaluation, Grad-CAM visualization, and performance analysis for reproducible research in explainable medical image segmentation.

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

> **"Making Medical AI Accurate, Transparent, and Trustworthy through Explainable Deep Learning."**
