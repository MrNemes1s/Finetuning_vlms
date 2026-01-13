#!/bin/bash
# Quick setup script for Runpod environment

set -e  # Exit on error

echo "======================================================================"
echo "Setting up Financial Document VLM Fine-tuning Environment"
echo "======================================================================"
echo ""

# Check if running with GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠ WARNING: nvidia-smi not found. Make sure you're running on a GPU instance."
else
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Update system packages
echo "Step 1: Updating system packages..."
apt-get update -qq

# Install system dependencies
echo "Step 2: Installing system dependencies..."
apt-get install -y -qq poppler-utils  # Required for pdf2image

# Install Python packages
echo "Step 3: Installing Python packages..."
pip install --upgrade pip -q

echo "  Installing core ML libraries..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q

echo "  Installing transformers and PEFT..."
pip install transformers accelerate peft bitsandbytes datasets evaluate -q

echo "  Installing PDF processing libraries..."
pip install pdf2image PyPDF2 Pillow -q

echo "  Installing Qwen2-VL dependencies..."
pip install qwen-vl-utils sentencepiece protobuf -q

echo "  Installing utilities..."
pip install scikit-learn numpy tqdm tensorboard -q

echo "  Installing Jupyter..."
pip install jupyter ipykernel ipywidgets -q

# Verify installations
echo ""
echo "Step 4: Verifying installations..."

python3 << EOF
import sys
import torch
import transformers
from peft import LoraConfig
from pdf2image import convert_from_path

print("✓ PyTorch version:", torch.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✓ CUDA version:", torch.version.cuda)
    print("✓ GPU device:", torch.cuda.get_device_name(0))
    print("✓ GPU memory:", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("✓ Transformers version:", transformers.__version__)
print("✓ All dependencies installed successfully!")
EOF

# Create directory structure
echo ""
echo "Step 5: Creating directory structure..."
mkdir -p data processed_pdfs finetuned_qwen2vl test_data logs

# Generate sample files
echo ""
echo "Step 6: Generating sample configuration files..."
python3 generate_sample_data.py

# Setup Jupyter kernel
echo ""
echo "Step 7: Setting up Jupyter kernel..."
python3 -m ipykernel install --user --name qwen2vl --display-name "Qwen2-VL FineTune"

# Set environment variables
echo ""
echo "Step 8: Setting environment variables..."
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE=./cache/transformers
export HF_HOME=./cache/huggingface

# Save environment variables
cat >> ~/.bashrc << 'EOF'
# Qwen2-VL Fine-tuning Environment
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE=./cache/transformers
export HF_HOME=./cache/huggingface
EOF

echo ""
echo "======================================================================"
echo "SETUP COMPLETE!"
echo "======================================================================"
echo ""
echo "Environment Details:"
echo "  - Python: $(python3 --version)"
echo "  - PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  - CUDA: $(python3 -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "Not available")')"
echo ""
echo "Next Steps:"
echo "  1. Start Jupyter Lab:"
echo "     jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo ""
echo "  2. Open the notebook:"
echo "     financial_document_vlm_finetuning.ipynb"
echo ""
echo "  3. Place your PDF files in the ./data/ directory"
echo ""
echo "  4. Update paths in the notebook and run!"
echo ""
echo "For monitoring GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "For viewing training logs:"
echo "  tensorboard --logdir=./finetuned_qwen2vl --host=0.0.0.0"
echo ""
