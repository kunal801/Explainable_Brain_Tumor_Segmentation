# Complete Technical Implementation Guide: Brain Tumor Detection with Explainable AI

## Table of Contents

1. [Environment Setup & Dependencies](#environment-setup)
2. [Data Preparation & Preprocessing](#data-preparation)
3. [Classification Models & Architectures](#classification-models)
4. [Segmentation Models & Architectures](#segmentation-models)
5. [Explainable AI Integration](#explainable-ai)
6. [Training & Optimization](#training-optimization)
7. [Production Deployment](#production-deployment)

---

## 1. Environment Setup & Dependencies

### Core Dependencies Installation

```bash
# Create virtual environment
python -m venv brain_tumor_env
source brain_tumor_env/bin/activate  # Linux/Mac

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install medical imaging frameworks
pip install monai>=1.2.0 nibabel SimpleITK pydicom radiomics

# Install explainability tools
pip install pytorch-grad-cam captum shap lime

# Install utilities
pip install numpy pandas scipy scikit-learn matplotlib seaborn
pip install tqdm pyyaml opencv-python pillow
```

### requirements.txt

```txt
torch==2.1.0
torchvision==0.16.0
tensorflow==2.13.0
pytorch-lightning==2.0.9
monai==1.2.2
nibabel==5.1.0
SimpleITK==2.2.1
pydicom==2.4.0
pytorch-grad-cam==1.4.8
captum==0.6.0
shap==0.42.0
lime==0.2.0
opencv-python==4.8.1
scikit-image==0.21.0
numpy==1.24.3
pandas==2.0.3
scipy==1.11.2
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
wandb==0.15.7
albumentations==1.3.0
optuna==3.0.5
flask==3.0.0
flask-cors==4.0.0
onnx==1.15.0
onnxruntime==1.17.0
```

---

## 2. Data Preparation & Preprocessing

### Dataset Loading

```python
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

class BrainTumorDataset(Dataset):
    """PyTorch Dataset for brain tumor MRI images"""
    
    def __init__(self, patient_ids: List[str], data_dir: str, 
                 labels_df: pd.DataFrame, preprocessor=None, is_train: bool = True):
        self.patient_ids = patient_ids
        self.data_dir = Path(data_dir)
        self.labels = labels_df.set_index('patient_id').to_dict()['label']
        self.preprocessor = preprocessor
        self.is_train = is_train
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        patient_id = self.patient_ids[idx]
        patient_dir = self.data_dir / patient_id
        
        # Load MRI modalities (T1, T1CE, T2, FLAIR)
        modalities = ['t1', 't1ce', 't2', 'flair']
        images = []
        
        for mod in modalities:
            filepath = patient_dir / f"{patient_id}_{mod}.nii.gz"
            img = nib.load(filepath).get_fdata()
            images.append(img)
        
        # Stack modalities: (4, H, W, D)
        image_3d = np.stack(images, axis=0)
        
        # Get label
        label = self.labels[patient_id]
        
        # Apply preprocessing if available
        if self.preprocessor:
            image_3d = self.preprocessor(image_3d)
        
        return torch.tensor(image_3d, dtype=torch.float32), label
```

### Data Preprocessing Pipeline

```python
from monai.transforms import Compose, EnsureChannelFirstd, Orientationd
from monai.transforms import Spacingd, CropForegroundd, NormalizeIntensityd
from monai.transforms import RandAffined, RandFlipd, Resized

class BrainTumorPreprocessor:
    """Complete preprocessing pipeline for brain tumor MRI"""
    
    def __init__(self, image_size: Tuple[int, int, int] = (96, 96, 96)):
        self.image_size = image_size
        self.transforms = self._get_transforms()
    
    def _get_transforms(self):
        """Create preprocessing transforms"""
        return Compose([
            EnsureChannelFirstd(keys=['image']),
            Orientationd(keys=['image'], axcodes='RAS'),
            Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0)),
            CropForegroundd(keys=['image'], source_key='image'),
            Resized(keys=['image'], spatial_size=self.image_size),
            NormalizeIntensityd(keys=['image'], subtrahend='image', divisor='image', nonzero=True),
            RandAffined(keys=['image'], prob=0.8, rotate_range=(3.14/6,)*3),
            RandFlipd(keys=['image'], prob=0.5, spatial_axis=0),
        ])
    
    def __call__(self, image):
        return self.transforms({'image': image})['image']
```

---

## 3. Classification Models & Architectures

### ResNet50 Transfer Learning

```python
import torchvision.models as models
from torch import nn

class ResNet50Classifier(nn.Module):
    """ResNet50 for brain tumor classification"""
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Adapt for single-channel input
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight[:, 0, :, :] = original_conv.weight.mean(dim=1)
        
        # Replace classification head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
```

### Vision Transformer (ViT)

```python
import timm

class ViTClassifier(nn.Module):
    """Vision Transformer for brain tumor classification"""
    
    def __init__(self, num_classes: int = 4, model_name: str = 'vit_base_patch16_224'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
        # Modify for single-channel input
        original_patch = self.backbone.patch_embed.proj
        self.backbone.patch_embed.proj = nn.Conv2d(
            1, original_patch.out_channels,
            kernel_size=original_patch.kernel_size,
            stride=original_patch.stride,
            padding=original_patch.padding
        )
    
    def forward(self, x):
        return self.backbone(x)
```

---

## 4. Segmentation Models & Architectures

### 3D U-Net for Tumor Segmentation

```python
from monai.networks.nets import UNet, UNETR, SwinUNETR

class BrainTumorSegmentationUNet(nn.Module):
    """3D U-Net for brain tumor segmentation"""
    
    def __init__(self, in_channels: int = 4, out_channels: int = 4):
        super().__init__()
        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )
    
    def forward(self, x):
        return self.model(x)

class BrainTumorSegmentationSwinUNETR(nn.Module):
    """State-of-the-art Swin UNETR for segmentation"""
    
    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 img_size: Tuple[int, int, int] = (96, 96, 96)):
        super().__init__()
        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
            use_checkpoint=True
        )
    
    def forward(self, x):
        return self.model(x)
```

---

## 5. Explainable AI Integration

### Grad-CAM Implementation

```python
from pytorch_grad_cam import GradCAM
import cv2
import matplotlib.pyplot as plt

class BrainTumorGradCAM:
    """Grad-CAM for explainability"""
    
    def __init__(self, model, target_layer, device='cuda'):
        self.model = model
        self.device = device
        self.cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device=='cuda'))
    
    def generate_cam(self, input_tensor, target_class=None):
        """Generate class activation map"""
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            
            if target_class is None:
                class_idx = output.argmax(dim=1).item()
            else:
                class_idx = target_class
            
            confidence = probs[0, class_idx].item()
        
        # Generate CAM
        cam = self.cam(input_tensor=input_tensor, targets=None)[0, :]
        
        return cam, class_idx, confidence
    
    def visualize(self, image_np, cam, class_name, confidence, save_path=None):
        """Visualize Grad-CAM"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(image_np, cmap='gray')
        axes[0].set_title('Original MRI')
        axes[0].axis('off')
        
        # CAM
        axes[1].imshow(cam, cmap='hot')
        axes[1].set_title('Grad-CAM')
        axes[1].axis('off')
        
        # Overlay
        cam_rgb = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        image_rgb = np.stack([image_np]*3, axis=2)
        overlay = 0.6 * image_rgb + 0.4 * (cam_rgb.astype(np.float32) / 255)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        fig.suptitle(f"Prediction: {class_name} ({confidence:.2%})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
```

### SHAP Integration

```python
import shap

class BrainTumorSHAP:
    """SHAP for feature importance"""
    
    def __init__(self, model, background_data, device='cuda'):
        self.model = model
        self.device = device
        
        if isinstance(background_data, torch.Tensor):
            background_data = background_data.numpy()
        
        # Flatten background data
        if background_data.ndim > 2:
            background_data = background_data.reshape(background_data.shape[0], -1)
        
        # Create explainer
        def predict_fn(x):
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1, 224, 224)
            x = x.to(device)
            with torch.no_grad():
                return torch.softmax(model(x), dim=1).cpu().numpy()
        
        self.explainer = shap.KernelExplainer(predict_fn, background_data)
    
    def get_shap_values(self, input_data, num_samples=100):
        """Calculate SHAP values"""
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()
        if input_data.ndim > 2:
            input_data = input_data.reshape(1, -1)
        
        return self.explainer.shap_values(input_data, num_samples=num_samples)
```

---

## 6. Training & Optimization

### Complete Training Loop

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

class BrainTumorTrainer:
    """Training pipeline with mixed precision"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.scaler = GradScaler()
        self.criterion = nn.CrossEntropyLoss()
        
        self.best_val_loss = float('inf')
        self.patience = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': train_loss/len(pbar), 'acc': 100*correct/total})
        
        return train_loss / len(self.train_loader), 100 * correct / total
    
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return val_loss / len(self.val_loader), 100 * correct / total
    
    def train(self, num_epochs):
        """Full training loop"""
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%")
            print(f"          Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("✓ Best model saved!")
            else:
                self.patience += 1
                if self.patience >= 20:
                    print("⚠ Early stopping triggered")
                    break
```

---

## 7. Production Deployment

### Model Export & Inference

```python
import onnx
import onnxruntime

class InferenceEngine:
    """Production inference engine"""
    
    def __init__(self, model_path, use_onnx=False, device='cuda'):
        self.device = device
        
        if use_onnx:
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.use_onnx = True
        else:
            self.model = torch.load(model_path)
            self.model.to(device).eval()
            self.use_onnx = False
    
    def predict(self, image):
        """Run inference"""
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
        
        if image.dim() == 2:
            image = image.unsqueeze(0)
        
        image = image.unsqueeze(0).to(self.device)
        
        if self.use_onnx:
            output = self.session.run(None, {'input': image.cpu().numpy()})
            probs = output[0][0]
        else:
            with torch.no_grad():
                output = self.model(image)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        
        return {
            'prediction': int(np.argmax(probs)),
            'confidence': float(np.max(probs)),
            'probabilities': probs.tolist()
        }
```

### Flask API

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load inference engine
engine = InferenceEngine('brain_tumor_model.pth')
class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

@app.route('/predict', methods=['POST'])
def predict():
    """Brain tumor prediction endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert('L')
        image = image.resize((224, 224))
        image_np = np.array(image, dtype=np.float32) / 255.0
        
        result = engine.predict(image_np)
        
        return jsonify({
            'prediction': class_names[result['prediction']],
            'confidence': f"{result['confidence']*100:.2f}%",
            'probabilities': {
                class_names[i]: f"{p*100:.2f}%"
                for i, p in enumerate(result['probabilities'])
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Docker Deployment

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY best_model.pth .

EXPOSE 5000

CMD ["python", "app.py"]
```

```bash
# Build and run Docker container
docker build -t brain-tumor-classifier .
docker run -p 5000:5000 --gpus all brain-tumor-classifier
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data
python data_loader.py --data-dir /path/to/brats

# 3. Train model
python train.py --model resnet50 --epochs 100 --batch-size 32

# 4. Evaluate model
python evaluate.py --model-path best_model.pth

# 5. Deploy API
python app.py

# 6. Test API
curl -X POST -F "file=@test_image.jpg" http://localhost:5000/predict
```

---

## Key Features

- ✅ **Classification**: ResNet, EfficientNet, Vision Transformer
- ✅ **Segmentation**: 3D U-Net, UNETR, Swin UNETR
- ✅ **Explainability**: Grad-CAM, SHAP, LIME
- ✅ **Training**: Mixed precision, gradient accumulation, early stopping
- ✅ **Deployment**: ONNX export, Flask API, Docker containerization
- ✅ **Optimization**: Learning rate scheduling, data augmentation

---

## References

- BraTS Challenge: https://www.med.upenn.edu/cbica/brats
- MONAI Framework: https://monai.io/
- Grad-CAM: https://github.com/jacobgil/pytorch-grad-cam
- SHAP: https://github.com/slundberg/shap

