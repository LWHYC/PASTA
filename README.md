# PASTA
A Data-Efficient Pan-Tumor Foundation Model for Oncology CT Interpretation

[Paper](https://arxiv.org/abs/2502.06171); [Dataset](https://huggingface.co/datasets/LWHYC/PASTA-Gen-30K); [Code](https://github.com/LWHYC/PASTA?tab=readme-ov-file)


# Overview
PASTA (Pan-Tumor Analysis with Synthetic Training Augmentation) is a data-efficient foundation model for analyzing diverse tumor lesions in 3D CT scans. Leveraging PASTA-Gen-30K, a large-scale synthetic dataset of 30,000 CT volumes with precise lesion masks and structured textual reports, PASTA addresses the scarcity of high-quality annotated data that traditionally hinders radiological AI research.

PASTA achieves state-of-the-art results on a wide range of tasks, including:

- Lesion segmentation
- Tumor detection in plain CT
- Tumor staging
- Survival prediction
- Structured report generation
- Cross-modality transfer learning

<img src="https://github.com/LWHYC/PASTA/blob/main/fig/fig1.png" alt="PASTA Logo" width="900" />
Workflow of PASTA Model Development and Training Pipeline. a, Overview of organs and lesion
types involved in PASTA training. b, Examples of lesions generated by PASTA-Gen from healthy organs. c,
Lesion generation process pipeline of PASTA-Gen. d, Two-stage training of PASTA using the PASTA-Gen-30K dataset.


# Key Features
1. **Synthetic Data Backbone**
Relies on PASTA-Gen-30K for training, bypassing the privacy constraints and annotation burdens associated with real clinical data.
2. **Data-Efficient**
Excels in few-shot settings, requiring only a small set of real-world annotated scans to reach high-performance levels.
3. **Pan-Tumor Coverage**
Encompasses malignancies across ten organ systems and five benign lesion types, designed for broad oncology analysis.

# Main results
<img src="https://github.com/LWHYC/PASTA/blob/main/fig/fig3.png" alt="PASTA Logo" width="900" />

Comparison on Lesion Segmentation. a, Example images of lesion segmentation results from various models. b, Comparison of model performance in lesion segmentation with sufficient data, measured by Dice Similarity Coefficients (DSC). c, Lesion segmentation performance of models under few-shot settings, with blue dashed lines indicating PASTA's full-data training results. Error bands denote 95% confidence intervals.

<img src="https://github.com/LWHYC/PASTA/blob/main/fig/fig4.png" alt="PASTA Logo" width="900" />

Performance on Various Oncological Tasks. a, Workflow of the classification tasks: target patches are cropped and passed through the encoder, followed by an MLP head to predict class probabilities. For tumor detection in plain CT scans (b, c), the target patch corresponds to the organ of interest, while for survival prediction and tumor staging tasks (e), the target patch is centered around the tumor region. b, c, Tumor identification performance of accuracy (b) and AUC (c) of models on Plain-CT data. d, Tumor segmentation performance (DSC) on plain-CT data. e, Performance of models in tumor staging and survival prediction across various tumor types. Bars in b and d plot displaying 95% confidence intervals as error bands.

<img src="https://github.com/LWHYC/PASTA/blob/main/fig/fig5.png" alt="PASTA Logo" width="900" />

Comparison on Structured Lesion Report Generation. a, Example of real and predicted lesion structure reports for bone metastasis generated by PASTA. b, Composition of the structured lesion report dataset, including 10 malignant lesion types (LuT: lung tumor, LiC: liver cancer, GC: gallbladder cancer, PT: pancreas tumor, EC: esophagus cancer, GC: gastric cancer, CC: colon cancer, KT: kidney tumor, BC: bladder cancer, and BM: bone metastasis) and 5 benign lesion types (LC: liver cyst, GS: gallstone, PC: pancreas cyst, KC: kidney cyst, and kidney stone). c, Comparison of Accuracy (ACC) and F1-scores for five structured report attributes across different models. Error bands denote 95% confidence intervals.

# Dataset
- Pretraining dataset PASTA-Gen-30K is available at [Hugging Face](https://huggingface.co/datasets/LWHYC/PASTA-Gen-30K).
    - Each synthetic 3D CT volume includes pixel-level lesion annotations and a structured radiological report.

# Checkpoint
- [Google Drive](https://drive.google.com/file/d/1A_PjIAqKg0y_Z986HSfTsYKLhc99EMkD/view?usp=drive_link)


# Train

**First of all, please standardize your dataset with:**
   
    python preprocess/NifitiStandard.py -in /path/to/original/data/root -out /path/to/save/data/root


## Segmentation
Prepare your dataset in the [this](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) format.

\
Finetuning with the following command: 
    
    python segmentation/nnunetv2/run/run_finetuning_pasta.py 3d_fullres PASTATrainer TASKID FOLD -pretrained_weights MODEL

Few-shot training with the following command: 
    
    python segmentation/nnunetv2/run/run_finetuning_pasta.py 3d_fullres PASTATrainer_fewshot TASKID FOLD -pretrained_weights MODEL


<!-- ## Classification -->
<!-- For the plain-CT tumor detection task and the structured report generation task, we use the [nnClassify](https://github.com/MIC-DKFZ/nnClassify) framework. -->

# Acknowledgement

- We thank the authors of [nnUNet](https://github.com/MIC-DKFZ/nnUNet), [STU-Net](https://github.com/uni-medical/STU-Net), [FMCIB](https://github.com/AIM-Harvard/foundation-cancer-image-biomarker) for their great works. Please cite their papers if you use our code.

```bibtex
@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}

@article{huang2023stu,
  title={Stu-net: Scalable and transferable medical image segmentation models empowered by large-scale supervised pre-training},
  author={Huang, Ziyan and Wang, Haoyu and Deng, Zhongying and Ye, Jin and Su, Yanzhou and Sun, Hui and He, Junjun and Gu, Yun and Gu, Lixu and Zhang, Shaoting and others},
  journal={arXiv preprint arXiv:2304.06716},
  year={2023}
}

@article{pai2024foundation,
  title={Foundation model for cancer imaging biomarkers},
  author={Pai, Suraj and Bontempi, Dennis and Hadzic, Ibrahim and Prudente, Vasco and Soka{\v{c}}, Mateo and Chaunzwa, Tafadzwa L and Bernatz, Simon and Hosny, Ahmed and Mak, Raymond H and Birkbak, Nicolai J and others},
  journal={Nature machine intelligence},
  volume={6},
  number={3},
  pages={354--367},
  year={2024},
  publisher={Nature Publishing Group UK London}
}