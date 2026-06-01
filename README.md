# Head & Neck Tumor Segmentation (Deep Learning MRI)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C)
![MMSegmentation](https://img.shields.io/badge/MMSegmentation-1.2.2-green)

## 📌 Clinical Overview
Automated segmentation of primary tumor volumes ($GTV_p$) and metastatic lymph nodes ($GTV_n$) in Head and Neck Cancer (HNC) is a critical bottleneck in radiotherapy (RT) planning. Due to poor soft-tissue contrast and complex anatomy, manual contouring is heavily subject to inter-observer variability. 

This repository provides an end-to-end automated Deep Learning pipeline to segment $GTV_p$ and $GTV_n$ on T2-weighted MRI scans (pre-RT and mid-RT), acting as a proof-of-concept for MRI-guided radiotherapy workflows.

## 🏗️ Architecture & Pipeline

The project is structured as a complete medical imaging pipeline, handling everything from raw NIfTI volumes to 3D morphological post-processing.

![Pipeline Architecture](assets/pipeline.png) 
*> Placeholder: Insert Figure 1 (Workflow Generale) from your report here.*

### 1. Data Stratification & Extraction
* **Dataset:** 120 unique patients (pre-RT and mid-RT scans).
* **Stratification:** Balanced split (Train 68%, Val 32%, Test 20%) maintaining clinical variance based on tumor presence and volumetric size.
* **Slicing:** Extraction of 2D axial slices with a $\pm 5$ slice contextual buffer to ensure spatial coherence.

### 2. Image Pre-processing (Enhancement Engine)
MRI T2 scans suffer from Rician noise and ambiguous tissue boundaries. The following pipeline is applied to every slice:
1. **Min-Max Normalization** to [0, 255] uint8 format.
2. **Custom Rician Denoising:** Utilizing Non-Local Means (NLM) coupled with Otsu's thresholding on a morphological gradient to clean homogeneous areas while preserving high-frequency edges.
3. **Median Filtering** ($3\times3$) to suppress salt-and-pepper artifacts.
4. **Edge Enhancement:** Sobel operators isolate biological contours, amplifying local contrast by a factor of 2.0 without boosting background noise.

![Pre-processing Steps](assets/preprocessing.png)
*> Placeholder: Insert Figure 3 (Effetto dell'applicazione delle tecniche...) showing the original vs Sobel+Contrast slices.*

### 3. Deep Learning Model
* **Architecture:** 2D U-Net implemented via `MMSegmentation`.
* **Transfer Learning:** Initialized with Cityscapes pre-trained weights to accelerate convergence on a limited medical dataset.
* **Optimization:** Batch Normalization (BN) utilized across the backbone and decode heads for stability.

![U-Net Architecture](assets/unet_architecture.png)
*> Placeholder: Insert Figure 2 (Struttura della U-Net) from your report here.*

### 4. Tackling Severe Class Imbalance (Composite Loss)
The dataset presents a critical imbalance: **98% background**, 1% $GTV_p$, and 1% $GTV_n$. Relying on standard Cross-Entropy leads to a collapsed model. I engineered a composite loss function:
* **Weighted CrossEntropyLoss (0.3):** Ensures stable gradient propagation.
* **Lovász-Softmax Loss (0.3):** Provides a differentiable surrogate to directly optimize the Intersection over Union (IoU) metric.
* **Tversky Loss (0.4):** Configured with $\alpha=0.7$ and $\beta=0.3$ to strictly penalize False Negatives (prioritizing clinical sensitivity over specificity).

### 5. 3D Morphological Post-Processing
2D predictions often lack spatial continuity. 2D masks are stacked back into NIfTI volumes and refined using 3D mathematical morphology:
* **Spurious Artifact Removal:** Filtering out isolated connected components (< 50 voxels for $GTV_p$, < 1200 voxels for $GTV_n$).
* **3D Morphological Closing:** Utilizing spherical structuring elements (radius 1 for $GTV_p$, radius 3 for $GTV_n$) to fill internal voids and regularize anatomical surfaces.

![Post-processing Results](assets/postprocessing.png)
*> Placeholder: Insert Figure 4 (or an animated GIF) showing Ground Truth vs Raw Prediction vs Post-processed Prediction.*

---

## 📊 Performance Baseline (Test Set)

While the complex and varied nature of HNC limits absolute accuracy on a basic 2D U-Net, the integration of targeted pre-processing and 3D post-processing yielded measurable improvements across the test cohort (24 patients).

| Metric | $GTV_p$ (Primary) | $GTV_n$ (Nodes) |
| :--- | :--- | :--- |
| **DSC (Pre-processing only)** | 0.262 ± 0.252 | 0.295 ± 0.237 |
| **DSC (+ 3D Post-processing)** | **0.269 ± 0.257** | **0.319 ± 0.275** |
| **$\Delta V\%$ Error (Auto vs Manual)** | 61.5% ± 79.4% | 38.6% ± 48.5% |

*Note: Results highlight the inherent difficulty of mid-RT morphological variations and the limitations of 2D architectures in capturing 3D spatial coherence, establishing a clear baseline for future 3D volumetric models (e.g., V-Net, 3D U-Net).*

---

## 🚀 Usage & Reproducibility

### 1. Environment Setup
Clone the repository and install the strict dependencies (requires CUDA 12.1).
```bash
git clone [https://github.com/yourusername/Head-Neck-Tumor-Segmentation.git](https://github.com/yourusername/Head-Neck-Tumor-Segmentation.git)
cd Head-Neck-Tumor-Segmentation
pip install -r requirements.txt
