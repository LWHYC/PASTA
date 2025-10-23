#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PASTA Installation Test Script
验证环境是否正确配置
"""

import sys
import os

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_success(text):
    print(f"✓ {text}")

def print_error(text):
    print(f"✗ {text}")

def print_warning(text):
    print(f"⚠ {text}")

def test_python_version():
    print_header("1. Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 9:
        print_success("Python version is compatible")
        return True
    else:
        print_error("Python 3.9+ is required")
        return False

def test_pytorch():
    print_header("2. PyTorch")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print_success("PyTorch imported successfully")
        
        if torch.cuda.is_available():
            print_success(f"CUDA is available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print_warning("CUDA is not available (CPU mode)")
        return True
    except ImportError as e:
        print_error(f"PyTorch import failed: {e}")
        return False

def test_medical_packages():
    print_header("3. Medical Image Processing Packages")
    packages = {
        'SimpleITK': 'SimpleITK',
        'nibabel': 'nibabel',
        'torchio': 'torchio'
    }
    
    all_ok = True
    for name, import_name in packages.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print_success(f"{name}: {version}")
        except ImportError as e:
            print_error(f"{name} import failed: {e}")
            all_ok = False
    
    return all_ok

def test_nnunetv2():
    print_header("4. nnUNetv2 (Local Version)")
    try:
        import nnunetv2
        file_path = nnunetv2.__file__
        print(f"nnUNetv2 location: {file_path}")
        
        # Check if it's the local version
        if "PASTA" in file_path and "segmentation" in file_path:
            print_success("Using local customized nnUNetv2 ✓")
            return True
        else:
            print_warning("nnUNetv2 is not from PASTA/segmentation folder")
            print("   Expected path should contain: 'PASTA/segmentation/nnunetv2'")
            return False
    except ImportError as e:
        print_error(f"nnUNetv2 import failed: {e}")
        print("   Please install: cd segmentation && pip install -e .")
        return False

def test_other_packages():
    print_header("5. Other Dependencies")
    packages = [
        'numpy', 'scipy', 'sklearn', 'tqdm', 'pandas', 
        'matplotlib', 'tensorboard'
    ]
    
    all_ok = True
    for pkg in packages:
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            print_success(f"{pkg}: {version}")
        except ImportError as e:
            print_error(f"{pkg} import failed: {e}")
            all_ok = False
    
    return all_ok

def test_environment_variables():
    print_header("6. nnUNet Environment Variables")
    env_vars = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
    
    all_set = True
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print_success(f"{var}: {value}")
        else:
            print_warning(f"{var} is not set")
            all_set = False
    
    if not all_set:
        print("\nTo set these variables:")
        print("  export nnUNet_raw=\"/path/to/nnUNet_raw\"")
        print("  export nnUNet_preprocessed=\"/path/to/nnUNet_preprocessed\"")
        print("  export nnUNet_results=\"/path/to/nnUNet_results\"")
    
    return all_set

def test_pythonpath():
    print_header("7. PYTHONPATH")
    pythonpath = os.environ.get('PYTHONPATH', '')
    
    if pythonpath:
        print(f"PYTHONPATH: {pythonpath}")
        if "PASTA" in pythonpath and "segmentation" in pythonpath:
            print_success("PYTHONPATH includes PASTA/segmentation")
            return True
        else:
            print_warning("PYTHONPATH doesn't include PASTA/segmentation")
            return False
    else:
        print_warning("PYTHONPATH is not set")
        print("   If using pip install -e, this is OK")
        return True

def main():
    print("\n" + "="*60)
    print("  PASTA Installation Test")
    print("="*60)
    
    results = {
        "Python Version": test_python_version(),
        "PyTorch": test_pytorch(),
        "Medical Packages": test_medical_packages(),
        "nnUNetv2": test_nnunetv2(),
        "Other Dependencies": test_other_packages(),
        "Environment Variables": test_environment_variables(),
        "PYTHONPATH": test_pythonpath()
    }
    
    print_header("Summary")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:30s} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("\n🎉 All tests passed! Your environment is ready to use PASTA.")
    else:
        print_warning(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print("\nFor detailed installation instructions, see INSTALL.md")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

