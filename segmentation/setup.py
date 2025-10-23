from setuptools import setup, find_packages

setup(
    name='nnunetv2-pasta',
    version='2.0.0',
    description='nnUNet v2 for PASTA project',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'SimpleITK>=2.2.0',
        'nibabel>=5.0.0',
        'batchgenerators>=0.25',
        'tqdm>=4.65.0',
        'pandas>=1.5.0',
        'scikit-learn>=1.2.0',
        'matplotlib>=3.7.0',
        'dicom2nifti>=2.4.0',
        'connected-components-3d>=3.10.0',
        'graphviz>=0.20',
    ],
)

