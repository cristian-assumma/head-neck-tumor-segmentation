# Head & Neck Tumor Segmentation (Deep Learning MRI)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C)
![MMSegmentation](https://img.shields.io/badge/MMSegmentation-1.2.2-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![Testing](https://img.shields.io/badge/PyTest-Passing-brightgreen)

## 📌 Clinical & Architectural Overview

Automated segmentation of primary tumor volumes ($GTV_p$) and metastatic lymph nodes ($GTV_n$) in Head and Neck Cancer (HNC) is a critical bottleneck in radiotherapy (RT) planning.

This repository provides an **End-to-End Deep Learning Infrastructure Proof-of-Concept**. While it implements a 2D U-Net baseline for segmentation on T2-weighted MRI scans, the primary focus of this project is the **software engineering of the clinical workflow**: automated data stratification, Rician noise enhancement, robust NIfTI/DICOM file handling, 3D morphological post-processing, and containerized deployment for reproducible inference.

## 🏗️ Architecture & Pipeline

The project is structured as a complete end-to-end medical imaging pipeline.

![Pipeline Architecture](docs/images/pipeline.png) 
*> Full workflow: from raw volumetric data to 3D morphological post-processing.*

### 1. Data Stratification & Extraction
* **Dataset:** 120 unique patients (pre-RT and mid-RT scans).
* **Stratification:** Balanced split (Train 68%, Val 32%, Test 20%) maintaining clinical variance based on tumor presence and volumetric size.
* **Slicing:** Extraction of 2D axial slices with a $\pm 5$ slice contextual buffer to ensure spatial coherence.

### 2. Image Pre-processing (Enhancement Engine)
MRI T2 scans suffer from Rician noise and ambiguous tissue boundaries. The pipeline applies:
1. **Min-Max Normalization** to [0, 255] uint8 format.
2. **Custom Rician Denoising:** Utilizing Non-Local Means (NLM) coupled with Otsu's thresholding on a morphological gradient to clean homogeneous areas while preserving high-frequency edges.
3. **Median Filtering** ($3\times3$) to suppress salt-and-pepper artifacts.
4. **Edge Enhancement:** Sobel operators isolate biological contours, amplifying local contrast by a factor of 2.0 without boosting background noise.

![Pre-processing Steps](docs/images/preprocessing.png)
*> Pre-processing steps applied sequentially to each slice.*

### 3. Deep Learning Model & Composite Loss
* **Architecture:** 2D U-Net implemented via `MMSegmentation`.
* **Transfer Learning:** Initialized with Cityscapes pre-trained weights to accelerate convergence on a limited medical dataset.
* **Optimization:** Batch Normalization (BN) utilized across the backbone and decode heads for stability.

![U-Net Architecture](docs/images/unet_architecture.png)
*> U-Net architecture used in this repo.*

### 4. Tackling Severe Class Imbalance (Composite Loss)
The dataset presents a critical imbalance: **98% background**, 1% $GTV_p$, and 1% $GTV_n$. Relying on standard Cross-Entropy leads to a collapsed model. I engineered a composite loss function:
* **Weighted CrossEntropyLoss (0.3):** Ensures stable gradient propagation.
* **Lovász-Softmax Loss (0.3):** Provides a differentiable surrogate to directly optimize the Intersection over Union (IoU) metric.
* **Tversky Loss (0.4):** Configured with $\alpha=0.7$ and $\beta=0.3$ to strictly penalize False Negatives (prioritizing clinical sensitivity over specificity).

### 5. 3D Morphological Post-Processing
2D predictions often lack spatial continuity. 2D masks are stacked back into NIfTI volumes and refined using 3D mathematical morphology:
* **Spurious Artifact Removal:** Filtering out isolated connected components (< 50 voxels for $GTV_p$, < 1200 voxels for $GTV_n$).
* **3D Morphological Closing:** Utilizing spherical structuring elements (radius 1 for $GTV_p$, radius 3 for $GTV_n$) to fill internal voids and regularize anatomical surfaces.

![Post-processing Results](docs/images/postprocessing.png)
*> Ground Truth vs Raw Prediction vs Post-processed Prediction.*

---

## 🛡️ Automated Testing & Quality Assurance

To improve software reliability and prevent silent regressions in volumetric calculations and metric evaluations, the core logic is covered by automated unit tests.

Run the test suite locally:

```bash
pytest tests/
```

### Covered Components

- NIfTI volume loading and reconstruction.
- File matching and metadata integrity.
- Morphological post-processing.
- Dice Score calculations.
- Inference pipeline utilities.

---

## 📊 Performance Baseline (Proof-of-Concept)

While the complex and varied nature of HNC limits absolute accuracy on a basic 2D U-Net, the integration of targeted pre-processing and 3D post-processing yielded measurable improvements across the test cohort (24 patients).

| Metric | $GTV_p$ (Primary) | $GTV_n$ (Nodes) |
| :--- | :--- | :--- |
| **DSC (Pre-processing only)** | 0.262 ± 0.252 | 0.295 ± 0.237 |
| **DSC (+ 3D Post-processing)** | **0.269 ± 0.257** | **0.319 ± 0.275** |

> **Note:** The current metrics reflect the inherent limitations of 2D architectures on highly complex 3D HNC morphological variations. This U-Net serves strictly as an architectural baseline to validate the data pipeline and deployment infrastructure.

In a production clinical setting, the inference engine is designed to seamlessly swap the baseline model with volumetric architectures such as:

- 3D U-Net
- V-Net
- MONAI-based segmentation networks

without requiring modifications to the surrounding workflow.

---

## 🚀 Usage & Reproducibility (Docker Deployment)

To eliminate dependency conflicts and ensure reproducibility across Linux, Windows, and macOS, the inference pipeline is fully containerized.

### 1. Build the Environment

Ensure Docker is installed, then build the image:

```bash
docker build -t hn-segmentation:v1 .
```

The container includes:

- PyTorch 2.1
- CUDA 12.1 support
- MMSegmentation
- MONAI dependencies
- Medical imaging libraries (SimpleITK, nibabel, etc.)

### 2. Quickstart Demo (Containerized Inference)

Download the required sample files (model weights and NIfTI volumes) and place them in the corresponding directories:

```text
models/
└── best_model.pth

data/
└── Dataset_gz/
```

Run inference:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  hn-segmentation:v1 \
  --dataset-dir /app/data/Dataset_gz \
  --excel-path /app/data/sample_test.xlsx \
  --output-dir /app/data/Inference_Results \
  --config src/training/unet_config.py \
  --checkpoint /app/models/best_model.pth
```

> **Windows PowerShell users:** replace `$(pwd)` with `${PWD}`.

The pipeline automatically:

1. Matches MRI volumes and metadata.
2. Processes 2D slices.
3. Performs model inference.
4. Reconstructs 3D masks.
5. Applies morphological closing.
6. Saves the final output as:

```text
SEG_<patient_id>_post.nii.gz
```

inside the results directory.

---

## 🧰 Technology Stack

| Category | Technologies |
|-----------|-------------|
| Deep Learning | PyTorch, MMSegmentation |
| Medical Imaging | SimpleITK, nibabel |
| Data Processing | NumPy, Pandas |
| Computer Vision | OpenCV, scikit-image |
| Testing | PyTest |
| Deployment | Docker |
| Version Control | Git, GitHub |

---

## 📬 Contact

**Ing. Cristian Assumma**  
*MSc Biomedical Engineer | AI Healthcare & MedTech*

* [LinkedIn](https://www.linkedin.com/in/cristian-assumma-08890b224)
* [GitHub](https://github.com/cristian-assumma)

---

