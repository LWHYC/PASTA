# Installation Notes for Developers

## Problem Analysis

This document explains the installation issues that users might encounter and how they were resolved.

### Issue 1: Missing Command-Line Tools

**Problem:**
After running `pip install -e .` in the `segmentation/` directory, the nnUNet commands (like `nnUNetv2_plan_and_preprocess`) were not available.

**Root Cause:**
The `setup.py` was missing the `entry_points` configuration. Without this, pip only installs the Python package but doesn't create executable command-line scripts.

**Solution:**
Added `entry_points` to `segmentation/setup.py`:
```python
entry_points={
    'console_scripts': [
        'nnUNetv2_plan_and_preprocess=nnunetv2.experiment_planning.plan_and_preprocess_entrypoints:plan_and_preprocess_entry',
        # ... other commands
    ],
}
```

### Issue 2: Incomplete Dependencies

**Problem:**
Users encountered `ModuleNotFoundError` for `blosc2` and import errors from `acvl_utils`.

**Root Cause:**
These dependencies were not declared in `setup.py` or `requirements.txt`, so they weren't automatically installed.

**Solution:**
- Added `acvl-utils` and `blosc2` to both `setup.py` and `requirements.txt`
- These are critical dependencies for nnUNet's preprocessing functionality

### Issue 3: NumPy 2.0 Compatibility

**Problem:**
Error: `ValueError: numpy.dtype size changed, may indicate binary incompatibility`

**Root Cause:**
- NumPy 2.0+ introduced ABI changes
- `blosc2` (a dependency of `acvl-utils`) was compiled against NumPy 1.x
- The original `setup.py` specified `numpy>=1.24.0` without an upper bound, allowing NumPy 2.0+ to be installed

**Solution:**
Changed numpy requirement to: `numpy>=1.24.0,<2.0.0`

### Issue 4: acvl-utils Version Too Old

**Problem:**
Error: `ImportError: cannot import name 'crop_to_bbox' from 'acvl_utils.cropping_and_padding.bounding_boxes'`

**Root Cause:**
- Older versions of `acvl-utils` (like 0.2.1) don't have the `crop_to_bbox` function
- This function was added in version 0.2.2+
- pip might install an older cached version if dependency resolution fails

**Solution:**
- Specify minimum version: `acvl-utils>=0.2.5`
- Force reinstall if needed: `pip install --force-reinstall --no-deps 'acvl-utils>=0.2.5'`

## Dependency Chain

```
PASTA
├── requirements.txt (base dependencies)
└── segmentation/
    └── setup.py (nnUNetv2 installation)
        ├── numpy<2.0 (⚠️ important: blosc2 compatibility)
        ├── acvl-utils>=0.2.5 (⚠️ important: needs crop_to_bbox function)
        │   └── blosc2 (requires numpy<2.0)
        └── entry_points (for CLI commands)
```

## Best Practices for Maintenance

### 1. Version Constraints

Always specify both lower AND upper bounds for critical dependencies:

```python
# Good
'numpy>=1.24.0,<2.0.0'

# Bad (can break in the future)
'numpy>=1.24.0'
```

### 2. Complete Dependency Declaration

Declare ALL dependencies explicitly, including transitive ones if they have specific requirements:

```python
install_requires=[
    'torch>=2.0.0',
    'numpy>=1.24.0,<2.0.0',
    'acvl-utils',  # Don't assume it will be installed automatically
    'blosc2',      # Explicit even though it's a transitive dependency
]
```

### 3. Entry Points

For any package that provides command-line tools, always include `entry_points`:

```python
entry_points={
    'console_scripts': [
        'command_name=module.submodule:function_name',
    ],
}
```

### 4. Python Version Constraints

Specify supported Python versions:

```python
python_requires='>=3.9,<3.12'  # Be specific about what you've tested
```

### 5. Testing Installation

Create a simple test script that users can run after installation:

```bash
#!/bin/bash
# test_installation.sh

echo "Testing Python imports..."
python -c "import nnunetv2; print('✓ nnunetv2 imported successfully')"

echo "Testing command-line tools..."
which nnUNetv2_plan_and_preprocess || echo "✗ nnUNetv2_plan_and_preprocess not found"

echo "Testing dependencies..."
python -c "import blosc2; print('✓ blosc2 imported successfully')"
python -c "import acvl_utils; print('✓ acvl_utils imported successfully')"
```

## Upgrading Dependencies

When upgrading dependencies:

1. **Check for breaking changes** in the changelog
2. **Test locally** before updating
3. **Update version constraints** conservatively
4. **Document compatibility** in README

Example workflow:
```bash
# Create a test environment
conda create -n pasta_test python=3.9
conda activate pasta_test

# Install with new dependency version
pip install 'numpy==2.0.0'  # Test new version
pip install -e segmentation/

# Run tests
python -m pytest tests/
# Or manual testing
python -c "import nnunetv2; nnunetv2.run_tests()"
```

## Common Issues and Solutions

### Issue: "command not found" after installation

**Diagnosis:**
```bash
# Check if command exists
ls $CONDA_PREFIX/bin/nnUNetv2*

# Check if package is installed
pip show nnunetv2-pasta
```

**Solution:**
```bash
# Reinstall in editable mode
cd segmentation && pip install -e . && cd ..

# Verify
which nnUNetv2_plan_and_preprocess
```

### Issue: Import errors after installation

**Diagnosis:**
```bash
# Check installed version
pip show package-name

# Check for version conflicts
pip check
```

**Solution:**
```bash
# Reinstall with specific version
pip install 'package-name==x.y.z'

# Or reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: NumPy compatibility errors

**Diagnosis:**
```bash
python -c "import numpy; print(numpy.__version__)"
```

**Solution:**
```bash
# Downgrade to numpy 1.x
pip install 'numpy<2.0'

# Verify
python -c "import numpy, blosc2; print('OK')"
```

### Issue: acvl-utils import errors

**Diagnosis:**
```bash
pip show acvl-utils
python -c "from acvl_utils.cropping_and_padding.bounding_boxes import crop_to_bbox"
```

**Solution:**
```bash
# Upgrade to acvl-utils 0.2.5+
pip install --force-reinstall --no-deps 'acvl-utils>=0.2.5'

# Verify
python -c "from acvl_utils.cropping_and_padding.bounding_boxes import crop_to_bbox; print('OK')"
```

## Checklist for New Releases

Before releasing a new version:

- [ ] All dependencies have version constraints
- [ ] `entry_points` are properly configured
- [ ] `requirements.txt` matches `setup.py` dependencies
- [ ] Python version compatibility is tested
- [ ] Installation instructions are up-to-date
- [ ] README includes troubleshooting section
- [ ] Test installation in a fresh conda environment
- [ ] Verify all command-line tools work
- [ ] Document any breaking changes

## Contact

If you encounter installation issues not covered here, please:
1. Check existing GitHub Issues
2. Provide full error messages and environment info
3. Include output of `pip list` and `python --version`

