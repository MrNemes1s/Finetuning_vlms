# Quick Start Guide

## Setup on Runpod (5 minutes)

### 1. Launch Runpod Instance
- **Minimum GPU**: RTX 3090 (24GB VRAM)
- **Recommended GPU**: A5000, A6000, or RTX 4090 (24GB+ VRAM)
- **Template**: PyTorch with CUDA 11.8+
- **Storage**: 50GB+ for models and data

### 2. Upload Files
```bash
# Upload to your Runpod instance:
# - financial_document_vlm_finetuning.ipynb
# - setup_runpod.sh
# - generate_sample_data.py
# - requirements.txt
```

### 3. Run Setup Script
```bash
bash setup_runpod.sh
```

This will:
- Install all dependencies
- Setup directory structure
- Generate sample files
- Configure Jupyter kernel

### 4. Start Jupyter
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### 5. Open Notebook
Open `financial_document_vlm_finetuning.ipynb` in your browser.

---

## Data Preparation (10 minutes)

### Option A: Use Sample Data (for testing)
```bash
python3 generate_sample_data.py
```

This creates:
- `sample_schema.json` - Example field definitions
- `sample_annotations.json` - Example training data
- `minimal_test_annotations.json` - Quick test set

### Option B: Prepare Your Own Data

#### 1. Create Schema File (`my_schema.json`)
```json
{
  "fields": [
    {"name": "invoice_number", "type": "string", "required": true},
    {"name": "date", "type": "string", "required": true},
    {"name": "total_amount", "type": "number", "required": true}
  ]
}
```

#### 2. Place PDFs in `./data/` directory
```bash
./data/
  ├── invoice_001.pdf
  ├── invoice_002.pdf
  └── ...
```

#### 3. Create Annotations File (`my_annotations.json`)

**Important**: First run the PDF processor to generate images, then create annotations:

```python
# In Jupyter notebook, run:
from pdf2image import convert_from_path
from PIL import Image
import os

# Process PDFs
pdf_processor = PDFProcessor(dpi=200)
pdf_processor.batch_process_pdfs(
    ["./data/invoice_001.pdf"],
    "./processed_pdfs"
)
```

Then create your annotations file:
```json
[
  {
    "image_file": "processed_pdfs/invoice_001/page_1.png",
    "source": "my_dataset",
    "fields": {
      "invoice_number": "INV-001",
      "date": "2024-01-15",
      "total_amount": 1234.56
    }
  }
]
```

---

## Training (Variable time based on data size)

### 1. Open Notebook
Open `financial_document_vlm_finetuning.ipynb`

### 2. Run Setup Cells (1-10)
Run cells 1-10 to import libraries and define classes.

### 3. Configure Training (Cell 2)
Modify `FineTuningConfig` if needed:

```python
@dataclass
class FineTuningConfig:
    # For 24GB GPU: use these settings
    load_in_4bit: bool = True
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3

    # For 40GB+ GPU: you can increase batch size
    # per_device_train_batch_size: int = 2
```

### 4. Run Training (Cell 12)
Uncomment and modify the training code:

```python
# Define your data paths
PDF_FILES = [
    "./data/invoice_001.pdf",
    "./data/invoice_002.pdf",
]

SCHEMA_PATH = "./my_schema.json"
ANNOTATIONS_FILE = "./my_annotations.json"
OUTPUT_DIR = "./finetuned_qwen2vl"

# Run complete pipeline
pipeline, eval_results = run_complete_pipeline(
    pdf_files=PDF_FILES,
    schema_path=SCHEMA_PATH,
    annotations_file=ANNOTATIONS_FILE,
    output_dir=OUTPUT_DIR
)
```

### 5. Monitor Training
```bash
# In a new terminal:
tensorboard --logdir=./finetuned_qwen2vl --host=0.0.0.0 --port=6006

# Monitor GPU:
watch -n 1 nvidia-smi
```

---

## Using the Fine-tuned Model

### Load Model
```python
from PIL import Image

# Load fine-tuned model
pipeline = FineTuningPipeline(config)
pipeline.load_finetuned_model("./finetuned_qwen2vl")

# Create evaluator
evaluator = ModelEvaluator(
    pipeline.model,
    pipeline.processor,
    pipeline.tokenizer
)
```

### Make Predictions
```python
# Load image
image = Image.open("test_invoice.png")

# Define fields to extract
fields = ["invoice_number", "date", "total_amount", "vendor_name"]

# Extract
result = evaluator.predict(image, fields)
print(result)
```

Output:
```json
{
  "invoice_number": "INV-2024-001",
  "date": "2024-01-15",
  "total_amount": 1234.56,
  "vendor_name": "ACME Corp"
}
```

---

## Troubleshooting

### OOM (Out of Memory) Error
```python
# In Cell 2, adjust config:
config.load_in_4bit = True  # Enable 4-bit quantization
config.per_device_train_batch_size = 1  # Reduce batch size
config.max_image_size = (768, 768)  # Reduce image size
config.lora_r = 8  # Reduce LoRA rank
```

### Model Not Loading
```bash
# Check HuggingFace access:
huggingface-cli login

# Or set token:
export HF_TOKEN="your_token_here"
```

### Poor Accuracy
1. **More training data**: Add 20+ diverse examples
2. **More epochs**: Increase `num_train_epochs` to 5-10
3. **Better annotations**: Ensure ground truth is accurate
4. **Higher quality images**: Use 200+ DPI for PDF conversion

### Slow Training
- This is normal for large models
- Expected: 1-2 minutes per step with 24GB GPU
- Monitor with: `watch -n 1 nvidia-smi`

---

## Performance Expectations

### Training Time (varies by dataset size)
- **10 samples**: ~30 minutes
- **50 samples**: ~2 hours
- **100 samples**: ~4 hours
- **500+ samples**: ~8-16 hours

### Accuracy Targets
- **Basic extraction**: 70-80% (after 3 epochs, 20 samples)
- **Good performance**: 85-90% (after 5 epochs, 50+ samples)
- **Production ready**: 95%+ (after 10 epochs, 100+ samples)

### GPU Memory Usage
- **4-bit quantization**: 8-10 GB
- **8-bit quantization**: 14-16 GB
- **Full precision**: 28+ GB

---

## File Structure After Training

```
docu_parse_trainer/
├── financial_document_vlm_finetuning.ipynb  # Main notebook
├── setup_runpod.sh                          # Setup script
├── generate_sample_data.py                  # Data generator
├── requirements.txt                         # Dependencies
├── README.md                                # Full documentation
├── QUICK_START.md                           # This file
│
├── data/                                    # Your PDFs
│   ├── invoice_001.pdf
│   └── ...
│
├── processed_pdfs/                          # Generated images
│   ├── invoice_001/
│   │   ├── page_1.png
│   │   └── ...
│   └── ...
│
├── finetuned_qwen2vl/                      # Output model
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── training_config.json
│   ├── evaluation_results.json
│   └── checkpoint-*/
│
└── logs/                                    # Training logs
    └── tensorboard/
```

---

## Next Steps

1. ✅ Setup environment
2. ✅ Prepare data
3. ✅ Train model
4. 🎯 **Deploy model** to production
5. 🎯 **Monitor performance** and iterate

## Need Help?

- Check README.md for detailed documentation
- Review notebook comments for inline help
- Check Qwen2-VL docs: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

---

## Tips for Success

1. **Start small**: Test with 5-10 samples first
2. **Validate data**: Ensure annotations are accurate
3. **Monitor training**: Watch loss curves in TensorBoard
4. **Iterate**: Start with 3 epochs, increase if needed
5. **Save checkpoints**: Keep multiple checkpoints for comparison

---

Good luck with your fine-tuning! 🚀
