# Explainable Brain Tumor Segmentation

> An explainable deep learning framework for automated brain tumor segmentation from MRI scans using EfficientNet-B2, U-Net, and Grad-CAM.

---

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

## Overview

Brain tumor segmentation plays a critical role in computer-aided diagnosis, treatment planning, and disease monitoring. Manual delineation of tumor boundaries from Magnetic Resonance Imaging (MRI) scans is labor-intensive, time-consuming, and subject to inter-observer variability.

This repository presents an end-to-end deep learning framework for automated brain tumor segmentation that combines an EfficientNet-B2 encoder with a U-Net decoder for accurate pixel-level localization. To improve transparency and clinical interpretability, Grad-CAM is integrated to generate visual explanations highlighting the regions that influence the model's predictions.

The framework is optimized using a hybrid Binary Cross-Entropy (BCE) and Dice Loss function, enabling robust learning on highly imbalanced medical imaging datasets while maintaining high segmentation accuracy.

---

## Key Features

- Automated brain tumor segmentation from MRI images
- EfficientNet-B2 encoder for hierarchical feature extraction
- U-Net decoder with skip connections for precise localization
- Grad-CAM based explainability
- Hybrid BCE + Dice Loss optimization
- Comprehensive evaluation metrics
- End-to-end reproducible training and evaluation pipeline

---

## System Architecture

```text
                 MRI Image
                     │
                     ▼
              Image Preprocessing
                     │
                     ▼
          EfficientNet-B2 Encoder
                     │
                     ▼
            Multi-scale Features
                     │
             Skip Connections
                     │
                     ▼
               U-Net Decoder
                     │
                     ▼
          Tumor Segmentation Mask
              │                │
              ▼                ▼
      Performance         Grad-CAM
       Evaluation      Visual Explanation
```

---

## Repository Structure

```text
Explainable-Brain-Tumor-Segmentation
│
├── dataset/
├── models/
├── training/
├── evaluation/
├── results/
├── paper/
├── presentation/
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

Clone the repository.

```bash
git clone https://github.com/<username>/Explainable-Brain-Tumor-Segmentation.git
cd Explainable-Brain-Tumor-Segmentation
```

Install the required packages.

```bash
pip install -r requirements.txt
```

---

## Dataset

The project uses the **Brain Tumor MRI Dataset: Segmentation and Classification**.

Dataset Statistics

| Property | Value |
|-----------|-------|
| Images | ~5,000 |
| Segmentation Masks | ~2,700 |
| Classes | Glioma, Meningioma, Pituitary, No Tumor |
| Input Resolution | 256 × 256 |

Expected directory structure:

```text
dataset/
├── train/
│   ├── images/
│   └── masks/
├── validation/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

---

## Training

Train the segmentation model.

```bash
python train.py
```

The training pipeline performs:

- Data preprocessing
- Model initialization
- Optimization using BCE + Dice Loss
- Checkpoint generation
- Metric logging

---

## Evaluation

Evaluate the trained model.

```bash
python evaluate.py
```

Generated outputs include:

- Dice Score
- IoU
- Precision
- Recall
- F1-Score
- Accuracy
- Segmentation masks
- Performance reports
- Grad-CAM visualizations

---

## Explainability

Grad-CAM is integrated into the inference pipeline to visualize the regions responsible for the segmentation output.

Generated visualizations include:

- Original MRI image
- Ground truth mask
- Predicted segmentation
- Grad-CAM heatmap
- Overlay visualization

---

## Model Configuration

| Parameter | Value |
|-----------|------:|
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

## Results

| Metric | Score |
|---------|------:|
| Dice Score | **0.9457** |
| IoU | **0.8986** |
| Precision | **0.9484** |
| Recall | **0.9445** |
| F1 Score | **0.9464** |
| Accuracy | **0.9692** |

The proposed EfficientNet-B2 U-Net framework achieves strong segmentation performance while maintaining model interpretability through Grad-CAM.

---

## Citation

```bibtex
@article{beniwal2026brainsegmentation,
  title={An Explainable Deep Learning Framework for Automated MRI Brain Tumor Segmentation},
  author={Beniwal, Kunal Singh and Sarkar, Arshia and Kansara, Shubham and Ghosh, Arpita},
  year={2026},
  institution={Vellore Institute of Technology}
}
```

---

## License

This project is released under the MIT License.

---

## Authors

**Kunal Singh Beniwal**

School of Computer Science and Engineering  
Vellore Institute of Technology

**Arshia Sarkar**

School of Computer Science and Engineering  
Vellore Institute of Technology

**Shubham Kansara**

School of Computer Science and Engineering  
Vellore Institute of Technology

**Dr. Arpita Ghosh**  
Project Supervisor  
School of Computer Science and Engineering  
Vellore Institute of Technology

---

## Acknowledgements

- Vellore Institute of Technology
- Kaggle Brain Tumor MRI Dataset
- EfficientNet
- U-Net
- Grad-CAM
- PyTorch
- TensorFlow
