# PASTA
A Data-Efficient Pan-Tumor Foundation Model for Oncology CT Interpretation


# Overview
PASTA (Pan-Tumor Analysis with Synthetic Training Assistance) is a data-efficient foundation model for analyzing diverse tumor lesions in 3D CT scans. Leveraging PASTA-Gen-30K, a large-scale synthetic dataset of 30,000 CT volumes with precise lesion masks and structured textual reports, PASTA addresses the scarcity of high-quality annotated data that traditionally hinders radiological AI research.

PASTA achieves state-of-the-art results on a wide range of tasks, including:

- Lesion segmentation
- Tumor detection in plain CT
- Tumor staging
- Survival prediction
- Structured report generation
- Cross-modality transfer learning

<img src="./PASTA/fig/fig1.png" alt="PASTA Logo" width="300" />



# Key Features
1. **Synthetic Data Backbone**
Relies on PASTA-Gen-30K for training, bypassing the privacy constraints and annotation burdens associated with real clinical data.
2. **Data-Efficient**
Excels in few-shot settings, requiring only a small set of real-world annotated scans to reach high-performance levels.
3. **Pan-Tumor Coverage**
Encompasses malignancies across ten organ systems and five benign lesion types, designed for broad oncology analysis.

# Dataset
- PASTA-Gen-30K is available at [Hugging Face](https://huggingface.co/datasets/LWHYC/PASTA-Gen-30K).
    - Each synthetic 3D CT volume includes pixel-level lesion annotations and a structured radiological report.

# Checkpoint
- 

# Train
- ## Segmentation
    - Our segmentation part is based on the [nnUNet](https://github.com/MIC-DKFZ/nnUNet). Many thanks to the authors for their great work.
    - Prepare your dataset in the [this](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) format.