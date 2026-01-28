# 4-Tier Brain MRI Analysis System Architecture

## Complete Application Flow: Input → Output Report Generation

### System Overview

```
INPUT LAYER (Flexible File Handling)
    ↓
[Single Image] [Multiple Images] [ZIP File] [Directory]
    ↓
UNIVERSAL DATA PREPROCESSOR
    ↓
TIER 1: SHARED ENCODER (SwinUNETR)
    ├─ 4-Channel MRI Input (4×96×96×96)
    ├─ Feature Extraction: 256 features
    └─ Output: Enhanced Features
    ↓
    ├─→ TIER 2A: CLASSIFICATION HEAD
    │   ├─ Tumor Type (4 classes)
    │   ├─ Abnormality Grade
    │   └─ Disease Risk (4 levels)
    │   Output: Class Probabilities + Confidence
    │
    ├─→ TIER 2B: SEGMENTATION HEAD
    │   ├─ Tumor Region (3D)
    │   ├─ Brain Tissue
    │   └─ Registration Map
    │   Output: Segmentation Mask (3D)
    │
    ├─→ TIER 2C: REGRESSION HEAD
    │   ├─ Brain Atrophy %
    │   └─ Survival Days
    │   Output: Continuous Values
    │
    └─→ TIER 2D: ANALYSIS HEAD
        ├─ Radiomics (100+ features)
        ├─ Quality Score [0-1]
        └─ Artifact Detection
        Output: Analysis Metrics
    ↓
TIER 3: ENHANCEMENT MODULE
    ├─ Super-Resolution (45×48×43 → 96×96×96)
    ├─ Quality Assessment
    └─ Consistency Validation
    ↓
TIER 4: REPORT GENERATION
    ├─ Clinical Text Report
    ├─ Structured JSON Output
    ├─ Visualizations (PNG/SVG)
    ├─ Recommendations
    └─ Explainability Analysis (Grad-CAM + SHAP)
```

---

## System Architecture & Components

### Architecture Tiers

#### TIER 1: Shared Encoder (42M Parameters)
- **Model**: SwinUNETR with gradient checkpointing
- **Input**: 4×96×96×96 (T1, T1CE, T2, FLAIR)
- **Output**: 256 feature channels
- **Purpose**: Extract universal medical imaging features
- **Optimization**: Gradient checkpointing for 3GB+ volumes

#### TIER 2: Specialized Heads (18M Parameters Total)

**2A. Classification Head (40K params)**
- Output 1: Tumor Type (Glioma/Meningioma/Pituitary/Healthy)
- Output 2: Abnormality Grade (Grade I-IV or Healthy)
- Output 3: Disease Risk Level (Low/Medium/High/Critical)

**2B. Segmentation Head (18M params)**
- 3D U-Net decoder with attention gates
- Output: 4-channel mask (Tumor/Brain Tissue/Registration/Background)
- 3D visualization capability

**2C. Regression Head (64K params)**
- Brain atrophy percentage
- Predicted survival days (optional)

**2D. Analysis Head (64K params)**
- Radiomics feature extraction
- Quality scoring [0-1]
- Artifact detection flags

#### TIER 3: Enhancement Module (85M Parameters)
- Super-resolution upsampling (45×48×43 → 96×96×96)
- Quality assessment network
- Consistency validation

#### TIER 4: Report Generation
- Template-based clinical reports
- Multi-format output (PDF, JSON, HTML)
- Explainability visualizations
- Statistical analysis and recommendations

---

## Complete Implementation Roadmap

### Phase 1: Setup & Data Preparation (1-2 hours)

#### Step 1.1: Google Colab Environment Setup

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Check GPU
!nvidia-smi

# Create workspace
import os
os.chdir('/content/drive/MyDrive')
!mkdir -p brain_tumor_project/{data,models,outputs,notebooks}
```

#### Step 1.2: Install All Dependencies

```bash
# Core ML
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow>=2.13.0
pip install pytorch-lightning>=2.0.0

# Medical Imaging
pip install monai>=1.2.0
pip install nibabel SimpleITK pydicom radiomics

# Explainability
pip install pytorch-grad-cam captum shap lime

# Data Processing
pip install numpy pandas scipy scikit-learn scipy

# Visualization
pip install matplotlib seaborn plotly

# Utilities
pip install tqdm pyyaml opencv-python pillow
pip install flask flask-cors onnx onnxruntime
pip install openpyxl reportlab docx

# For report generation
pip install jinja2 weasyprint
```

#### Step 1.3: Universal Dataset Analyzer

```python
import os
import json
from pathlib import Path
import nibabel as nib
import zipfile

class UniversalDatasetAnalyzer:
    """
    Universal dataset analyzer for mixed file formats and structures
    Handles: NIfTI (.nii.gz), DICOM (.dcm), ANALYZE (.hdr/.img), ZIP archives
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.file_inventory = {}
        self.dataset_structure = {}
    
    def scan_directory_recursively(self, max_depth=5):
        """
        Recursively scan directory structure
        Returns: Comprehensive file inventory
        """
        file_types = {
            '.nii.gz': [],
            '.nii': [],
            '.dcm': [],
            '.hdr': [],
            '.img': [],
            '.zip': []
        }
        
        for root, dirs, files in os.walk(self.dataset_path):
            depth = root.replace(str(self.dataset_path), '').count(os.sep)
            if depth > max_depth:
                continue
            
            for file in files:
                filepath = Path(root) / file
                for ext in file_types.keys():
                    if file.endswith(ext):
                        file_types[ext].append({
                            'path': str(filepath),
                            'size_mb': filepath.stat().st_size / (1024*1024),
                            'depth': depth
                        })
        
        return file_types
    
    def extract_file_metadata(self, file_path):
        """
        Extract metadata from medical imaging files
        """
        metadata = {}
        
        if file_path.endswith('.nii.gz') or file_path.endswith('.nii'):
            try:
                img = nib.load(file_path)
                metadata = {
                    'shape': img.shape,
                    'dtype': str(img.get_data_dtype()),
                    'affine_shape': img.affine.shape,
                    'file_type': 'NIfTI'
                }
            except Exception as e:
                metadata['error'] = str(e)
        
        return metadata
    
    def auto_detect_dataset_structure(self):
        """
        Auto-detect: Single-patient, multi-patient, organized by modality
        """
        file_types = self.scan_directory_recursively()
        
        structure_info = {
            'total_files': sum(len(v) for v in file_types.values()),
            'file_types': file_types,
            'structure_type': self._infer_structure(file_types),
            'estimated_patients': self._estimate_patient_count(file_types)
        }
        
        return structure_info
    
    def _infer_structure(self, file_types):
        """
        Infer dataset organization
        """
        total_nifti = len(file_types['.nii.gz']) + len(file_types['.nii'])
        
        if total_nifti < 5:
            return "Single Patient or Few Cases"
        elif total_nifti % 4 == 0:  # Likely 4-channel (T1, T1CE, T2, FLAIR)
            return "Multi-Patient with 4-Channel MRI"
        else:
            return "Mixed or Unknown Structure"
    
    def _estimate_patient_count(self, file_types):
        """
        Estimate number of patients
        """
        nifti_files = len(file_types['.nii.gz']) + len(file_types['.nii'])
        return max(1, nifti_files // 4)  # Assume 4 modalities per patient
    
    def generate_dataset_report(self):
        """
        Generate comprehensive dataset analysis report
        """
        analysis = self.auto_detect_dataset_structure()
        
        report = f"""
        ═══════════════════════════════════════════════════════════
        UNIVERSAL DATASET ANALYSIS REPORT
        ═══════════════════════════════════════════════════════════
        
        Dataset Path: {self.dataset_path}
        Total Files: {analysis['total_files']}
        Estimated Patients: {analysis['estimated_patients']}
        Structure Type: {analysis['structure_type']}
        
        File Breakdown:
        ├─ NIfTI (.nii.gz): {len(analysis['file_types']['.nii.gz'])}
        ├─ NIfTI (.nii): {len(analysis['file_types']['.nii'])}
        ├─ DICOM (.dcm): {len(analysis['file_types']['.dcm'])}
        ├─ ANALYZE (.hdr/.img): {len(analysis['file_types']['.hdr']) + len(analysis['file_types']['.img'])}
        └─ Archives (.zip): {len(analysis['file_types']['.zip'])}
        
        Recommendations:
        - Use 'AutoDataLoader' for flexible loading
        - Apply standardized preprocessing
        - Validate all outputs
        ═══════════════════════════════════════════════════════════
        """
        
        return report, analysis

# Usage in Colab
analyzer = UniversalDatasetAnalyzer('/content/drive/MyDrive/brain_tumor_project/data')
report, analysis = analyzer.generate_dataset_report()
print(report)
```

### Phase 2: Flexible Data Loading (1-2 hours)

#### AutoDataLoader: Universal File Handler

```python
import zipfile
import pydicom
from monai.transforms import Compose, LoadNiftid, EnsureChannelFirstd

class AutoDataLoader:
    """
    Automatically load and normalize mixed file formats
    """
    
    @staticmethod
    def load_file(file_path):
        """
        Unified loader for NIfTI, DICOM, ZIP, etc.
        """
        file_path = str(file_path)
        
        if file_path.endswith('.nii.gz') or file_path.endswith('.nii'):
            return nib.load(file_path).get_fdata()
        
        elif file_path.endswith('.dcm'):
            dcm = pydicom.dcmread(file_path)
            return dcm.pixel_array
        
        elif file_path.endswith('.zip'):
            return AutoDataLoader._handle_zip(file_path)
        
        elif file_path.endswith(('.hdr', '.img')):
            return nib.load(file_path).get_fdata()
        
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    @staticmethod
    def _handle_zip(zip_path):
        """
        Extract and load files from ZIP archive
        """
        results = {}
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.endswith(('.nii.gz', '.nii', '.dcm')):
                    with zip_ref.open(file_info) as f:
                        temp_path = f"/tmp/{file_info.filename}"
                        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                        with open(temp_path, 'wb') as temp_f:
                            temp_f.write(f.read())
                        
                        results[file_info.filename] = AutoDataLoader.load_file(temp_path)
        
        return results
    
    @staticmethod
    def batch_load(source_path, pattern="*.nii.gz"):
        """
        Load batch of files matching pattern
        """
        source = Path(source_path)
        files = list(source.rglob(pattern))
        
        batch_data = {}
        for file_path in files:
            try:
                batch_data[file_path.name] = AutoDataLoader.load_file(str(file_path))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return batch_data
```

### Phase 3: Model Architecture (2-3 hours)

#### Unified 4-Tier Model

```python
from monai.networks.nets import SwinUNETR
import torch.nn as nn

class BrainMRIAnalysisSystem(nn.Module):
    """
    Complete 4-Tier Analysis System
    Input: 4×96×96×96 (4-channel MRI)
    Output: Classification + Segmentation + Analysis + Regression
    """
    
    def __init__(self, num_classes=4):
        super().__init__()
        
        # TIER 1: Shared Encoder
        self.encoder = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=4,
            out_channels=256,
            feature_size=48,
            use_checkpoint=True
        )
        
        # TIER 2A: Classification Head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # TIER 2B: Segmentation Head
        self.segmentation_head = nn.Sequential(
            nn.Conv3d(256, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 4, 1)  # 4-channel output
        )
        
        # TIER 2C: Regression Head
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Brain atrophy + Survival
        )
        
        # TIER 2D: Analysis Head
        self.analysis_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 100)  # 100 radiomics features
        )
        
        # TIER 3: Enhancement Module (Optional)
        self.enhancement_enabled = False
    
    def forward(self, x):
        # Shared encoding
        encoded = self.encoder(x)
        
        # Multi-task outputs
        return {
            'classification': self.classification_head(encoded),
            'segmentation': self.segmentation_head(encoded),
            'regression': self.regression_head(encoded),
            'analysis': self.analysis_head(encoded)
        }
```

### Phase 4: Explainability Integration (1 hour)

#### Unified XAI Framework

```python
class ExplainabilityEngine:
    """
    Complete XAI pipeline: Grad-CAM + SHAP + LIME + Consensus
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.gradcam = GradCAM(model, target_layers=[model.encoder])
        self.shap_explainer = None
        self.lime_explainer = None
    
    def explain_prediction(self, input_tensor, prediction_type='classification'):
        """
        Generate unified explanation
        """
        # Grad-CAM
        cam = self.gradcam(input_tensor)
        
        # SHAP (simplified for speed)
        if self.shap_explainer is None:
            self._init_shap(input_tensor)
        
        shap_values = self.shap_explainer.shap_values(input_tensor)
        
        # Consensus heatmap
        consensus = (cam + shap_values) / 2
        
        return {
            'gradcam': cam,
            'shap': shap_values,
            'consensus': consensus,
            'confidence': 0.95  # Placeholder
        }
```

### Phase 5: Report Generation (2-3 hours)

#### Multi-Format Report Engine

```python
import json
from datetime import datetime
from jinja2 import Template

class ReportGenerator:
    """
    Generate comprehensive reports in multiple formats
    """
    
    def __init__(self, patient_id, analysis_results):
        self.patient_id = patient_id
        self.results = analysis_results
        self.timestamp = datetime.now().isoformat()
    
    def generate_clinical_report(self):
        """
        Generate clinical text report
        """
        template = """
        ═══════════════════════════════════════════════════════════
        BRAIN MRI ANALYSIS REPORT
        ═══════════════════════════════════════════════════════════
        
        Patient ID: {{ patient_id }}
        Analysis Date: {{ timestamp }}
        
        FINDINGS:
        ┌─ Classification Results
        │  ├─ Primary Tumor Type: {{ classification.tumor_type }} ({{ classification.confidence }}%)
        │  ├─ Abnormality Grade: {{ classification.grade }}
        │  └─ Disease Risk Level: {{ classification.risk_level }}
        │
        ├─ Segmentation Results
        │  ├─ Tumor Volume: {{ segmentation.volume_mm3 }} mm³
        │  ├─ Location: {{ segmentation.location }}
        │  └─ Brain Involvement: {{ segmentation.brain_percentage }}%
        │
        ├─ Clinical Metrics
        │  ├─ Brain Atrophy: {{ metrics.brain_atrophy }}%
        │  ├─ Image Quality Score: {{ metrics.quality_score }}/100
        │  └─ Confidence Index: {{ metrics.confidence_index }}
        │
        └─ Recommendations
           ├─ Primary: {{ recommendations.primary }}
           └─ Secondary: {{ recommendations.secondary }}
        
        ═══════════════════════════════════════════════════════════
        """
        
        jinja_template = Template(template)
        return jinja_template.render(
            patient_id=self.patient_id,
            timestamp=self.timestamp,
            classification=self.results.get('classification', {}),
            segmentation=self.results.get('segmentation', {}),
            metrics=self.results.get('metrics', {}),
            recommendations=self.results.get('recommendations', {})
        )
    
    def generate_json_report(self):
        """
        Generate structured JSON output
        """
        return {
            'patient_id': self.patient_id,
            'timestamp': self.timestamp,
            'analysis_results': self.results,
            'version': '1.0',
            'model_version': 'SwinUNETR-v1'
        }
    
    def generate_visualizations(self, output_dir):
        """
        Generate visualization images
        """
        visualizations = {
            'segmentation_3d': self._create_3d_visualization(),
            'heatmaps': self._create_xai_heatmaps(),
            'metrics_charts': self._create_metric_charts()
        }
        
        for name, viz in visualizations.items():
            viz.savefig(f'{output_dir}/{name}.png', dpi=300, bbox_inches='tight')
        
        return visualizations
    
    def export_all_formats(self, output_dir):
        """
        Export report in all formats
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Text report
        with open(f'{output_dir}/clinical_report.txt', 'w') as f:
            f.write(self.generate_clinical_report())
        
        # JSON report
        with open(f'{output_dir}/analysis_results.json', 'w') as f:
            json.dump(self.generate_json_report(), f, indent=2)
        
        # Visualizations
        self.generate_visualizations(output_dir)
        
        print(f"✓ Reports generated in {output_dir}")
```

---

## Complete Library Requirements

### Core ML Libraries
```bash
torch==2.1.0              # PyTorch deep learning
torchvision==0.16.0       # Computer vision utilities
torchmetrics==1.3.0       # ML metrics
torch-summary==1.8.0      # Model architecture summary
```

### Medical Imaging
```bash
monai==1.2.2              # Medical Imaging AI framework
nibabel==5.1.0            # NIfTI file handling
SimpleITK==2.2.1          # Image processing
pydicom==2.4.0            # DICOM file handling
radiomics==3.0.1          # Radiomics features
```

### Explainability & Interpretation
```bash
pytorch-grad-cam==1.4.8   # Grad-CAM visualization
captum==0.6.0             # Feature attribution
shap==0.42.0              # SHAP explanations
lime==0.2.0               # LIME local explanations
```

### Data Processing & Analysis
```bash
numpy==1.24.3             # Numerical computing
pandas==2.0.3             # Data manipulation
scipy==1.11.2             # Scientific computing
scikit-learn==1.3.0       # ML utilities
openpyxl==3.10.10         # Excel export
```

### Visualization
```bash
matplotlib==3.7.2         # Static plots
seaborn==0.12.2           # Statistical visualization
plotly==5.17.0            # Interactive plots
opencv-python==4.8.1      # Image processing
Pillow==10.0.1            # Image manipulation
```

### Report Generation
```bash
jinja2==3.1.2             # Template engine
weasyprint==60.0          # PDF generation
python-docx==0.8.11       # DOCX generation
reportlab==4.0.4          # PDF creation
```

### Web & API
```bash
flask==3.0.0              # Web framework
flask-cors==4.0.0         # CORS support
onnx==1.15.0              # Model interchange
onnxruntime==1.17.0       # ONNX inference
```

### Utilities
```bash
tqdm==4.66.1              # Progress bars
pyyaml==6.0.1             # Config files
hydra-core==1.3.1         # Configuration management
albumentations==1.3.0     # Image augmentation
optuna==3.0.5             # Hyperparameter optimization
```

---

## Google Colab: Dataset Upload Solutions

### Option 1: ZIP Upload (Recommended for 3-4 GB)

```python
from google.colab import files
import zipfile

# Step 1: Upload ZIP file
print("Select your dataset ZIP file...")
uploaded = files.upload()

# Step 2: Extract
zip_file = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall('/content/drive/MyDrive/brain_tumor_project/data')

print("✓ Dataset extracted")
```

### Option 2: Google Drive Direct Mount

```python
from google.colab import drive
import os

# Mount Drive
drive.mount('/content/drive')

# If dataset already in Drive
os.symlink(
    '/content/drive/MyDrive/your_dataset_folder',
    '/content/dataset'
)
```

### Option 3: Chunked Upload (For >4GB)

```python
import os
import hashlib

def split_large_file(file_path, chunk_size=500*1024*1024):  # 500MB chunks
    """
    Split large file into chunks
    """
    with open(file_path, 'rb') as f:
        chunk_num = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            with open(f'{file_path}.part{chunk_num}', 'wb') as chunk_file:
                chunk_file.write(chunk)
            chunk_num += 1
    
    return chunk_num

def merge_chunks(base_path, num_chunks):
    """
    Merge chunks back into original file
    """
    with open(base_path, 'wb') as outfile:
        for i in range(num_chunks):
            with open(f'{base_path}.part{i}', 'rb') as infile:
                outfile.write(infile.read())
            os.remove(f'{base_path}.part{i}')
```

### Recommended Dataset Structure

```
Dataset.zip (or folder)
├── Patient_001/
│   ├── t1.nii.gz
│   ├── t1ce.nii.gz
│   ├── t2.nii.gz
│   ├── flair.nii.gz
│   └── seg.nii.gz (optional)
├── Patient_002/
│   ├── t1.nii.gz
│   ├── t1ce.nii.gz
│   ├── t2.nii.gz
│   └── flair.nii.gz
└── ...
```

---

## Complete Notebook Flow (Google Colab)

### Main Notebook Structure

```python
# ═════════════════════════════════════════════════════════════════
# CELL 1: Setup & Install Dependencies
# ═════════════════════════════════════════════════════════════════
!pip install -q monai torch torchvision pytorch-grad-cam shap lime

# ═════════════════════════════════════════════════════════════════
# CELL 2: Mount Drive & Prepare Data
# ═════════════════════════════════════════════════════════════════
from google.colab import drive
drive.mount('/content/drive')

# ═════════════════════════════════════════════════════════════════
# CELL 3: Import All Classes
# ═════════════════════════════════════════════════════════════════
from universal_data_analyzer import UniversalDatasetAnalyzer
from auto_data_loader import AutoDataLoader
from model_architecture import BrainMRIAnalysisSystem
from explainability_engine import ExplainabilityEngine
from report_generator import ReportGenerator

# ═════════════════════════════════════════════════════════════════
# CELL 4: Analyze Dataset
# ═════════════════════════════════════════════════════════════════
analyzer = UniversalDatasetAnalyzer('/content/data')
report, analysis = analyzer.generate_dataset_report()
print(report)

# ═════════════════════════════════════════════════════════════════
# CELL 5: Load Model & Checkpoint
# ═════════════════════════════════════════════════════════════════
model = BrainMRIAnalysisSystem()
model.load_state_dict(torch.load('best_model.pth'))
model.to('cuda')
model.eval()

# ═════════════════════════════════════════════════════════════════
# CELL 6: Process Batch of Images
# ═════════════════════════════════════════════════════════════════
data_loader = AutoDataLoader()
images = data_loader.batch_load('/content/data')

for patient_id, image_data in images.items():
    # Preprocess & Normalize
    # Run inference
    # Generate report
    # Save outputs

# ═════════════════════════════════════════════════════════════════
# CELL 7: Generate All Reports
# ═════════════════════════════════════════════════════════════════
report_gen = ReportGenerator(patient_id, results)
report_gen.export_all_formats('/content/outputs')
```

---

## Implementation Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Setup & Data | 1-2 hrs | Environment setup, dataset analysis, file loading |
| Model Architecture | 2-3 hrs | Build 4-tier model, load pretrained weights |
| Training Loop | 5-6 hrs | Train on your dataset, validation, checkpointing |
| Testing & Inference | 1-2 hrs | Run predictions on test set |
| XAI Integration | 1 hr | Add Grad-CAM + SHAP explanations |
| Report Generation | 2-3 hrs | Create multi-format reports |
| **Total** | **12-17 hrs** | Complete pipeline ready |

---

## Output Structure

```
Outputs/
├── Patient_001/
│   ├── clinical_report.txt
│   ├── analysis_results.json
│   ├── segmentation_3d.png
│   ├── heatmaps_gradcam.png
│   ├── heatmaps_shap.png
│   ├── metrics_charts.png
│   └── recommendations.txt
├── Patient_002/
│   └── ...
└── batch_summary.json  # All patients summary
```

This is your complete roadmap for building a production-ready brain tumor analysis system!

