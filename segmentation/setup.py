from setuptools import setup, find_packages

setup(
    name='nnunetv2-pasta',
    version='2.0.0',
    description='nnUNet v2 for PASTA project - customized version',
    author='PASTA Team',
    packages=find_packages(),
    python_requires='>=3.9,<3.12',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0,<2.0.0',  # blosc2 not compatible with numpy 2.0+
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
        'acvl-utils>=0.2.5',  # Required for nnUNet preprocessing (>=0.2.5 for crop_to_bbox)
        'blosc2',  # Required for acvl-utils
    ],
    entry_points={
        'console_scripts': [
            'nnUNetv2_plan_and_preprocess=nnunetv2.experiment_planning.plan_and_preprocess_entrypoints:plan_and_preprocess_entry',
            'nnUNetv2_extract_fingerprint=nnunetv2.experiment_planning.plan_and_preprocess_entrypoints:extract_fingerprint_entry',
            'nnUNetv2_plan_experiment=nnunetv2.experiment_planning.plan_and_preprocess_entrypoints:plan_experiment_entry',
            'nnUNetv2_preprocess=nnunetv2.experiment_planning.plan_and_preprocess_entrypoints:preprocess_entry',
            'nnUNetv2_train=nnunetv2.run.run_training:run_training_entry',
            'nnUNetv2_predict=nnunetv2.inference.predict_from_raw_data:predict_entry_point',
        ],
    },
)






