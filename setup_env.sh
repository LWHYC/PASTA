#!/bin/bash
# PASTA Environment Setup Script

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== PASTA Environment Setup ===${NC}"

# Add local nnUNetv2 to PYTHONPATH
PASTA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PASTA_ROOT}/segmentation:${PYTHONPATH}"

echo -e "${GREEN}✓ Added local nnUNetv2 to PYTHONPATH${NC}"

# Setup nnUNet environment variables (you need to modify these paths)
if [ -z "$nnUNet_raw" ]; then
    echo -e "${YELLOW}⚠ nnUNet_raw not set. Please set it:${NC}"
    echo "  export nnUNet_raw=\"/path/to/nnUNet_raw\""
fi

if [ -z "$nnUNet_preprocessed" ]; then
    echo -e "${YELLOW}⚠ nnUNet_preprocessed not set. Please set it:${NC}"
    echo "  export nnUNet_preprocessed=\"/path/to/nnUNet_preprocessed\""
fi

if [ -z "$nnUNet_results" ]; then
    echo -e "${YELLOW}⚠ nnUNet_results not set. Please set it:${NC}"
    echo "  export nnUNet_results=\"/path/to/nnUNet_results\""
fi

echo -e "${GREEN}=== Setup Complete ===${NC}"
echo -e "Current PYTHONPATH: ${PYTHONPATH}"

