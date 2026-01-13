# Financial Document VLM Fine-tuning Module

A comprehensive Jupyter notebook for fine-tuning Qwen2-VL 7B on financial documents for key field extraction.

## Features

### 1. PDF Processing
- Convert PDF financial documents to images
- Automatic validation and preprocessing
- Batch processing support
- Configurable DPI and image sizing

### 2. Model Loading
- Qwen2-VL 7B with quantization support (4-bit/8-bit)
- Memory-efficient loading with BitsAndBytes
- Automatic device mapping for multi-GPU setups
- LoRA/QLoRA configuration for parameter-efficient fine-tuning

### 3. Data Preprocessing
- JSON schema validation for key fields
- Annotation validation against schema
- Train/eval/test split with configurable ratios
- Comprehensive data quality checks

### 4. Fine-tuning Pipeline
- LoRA adapter configuration
- Gradient accumulation for large effective batch sizes
- Mixed precision training (FP16)
- Automatic checkpointing and best model selection
- TensorBoard logging

### 5. Evaluation
- Field-level accuracy metrics
- Overall extraction accuracy
- Detailed per-sample results
- JSON output for further analysis

### 6. Model Management
- Save fine-tuned models with configurations
- Load and resume from checkpoints
- Easy deployment on inference servers

## Quick Start

### 1. Prepare Your Data

#### A. Create a schema JSON file (e.g., `schema.json`):
```json
{
  "fields": [
    {"name": "invoice_number", "type": "string", "required": true},
    {"name": "date", "type": "string", "required": true},
    {"name": "total_amount", "type": "number", "required": true},
    {"name": "vendor_name", "type": "string", "required": true},
    {"name": "tax_amount", "type": "number", "required": false},
    {"name": "currency", "type": "string", "required": false}
  ]
}
```

#### B. Create an annotations file (e.g., `annotations.json`):
```json
[
  {
    "image_file": "document1/page_1.png",
    "source": "training_set",
    "fields": {
      "invoice_number": "INV-2024-001",
      "date": "2024-01-15",
      "total_amount": 1234.56,
      "vendor_name": "ACME Corporation",
      "tax_amount": 123.45,
      "currency": "USD"
    }
  }
]
```

#### C. Organize your PDFs:
```
./data/
  ├── invoice1.pdf
  ├── invoice2.pdf
  └── ...
```

### 2. Run on Runpod

#### Step 1: Launch Runpod Instance
- Choose a GPU with at least 24GB VRAM (e.g., RTX 3090, A5000, or better)
- Select a PyTorch template with CUDA support
- Recommended: 40GB+ system RAM

#### Step 2: Upload Files
```bash
# Upload the notebook
# Upload your data directory
# Upload schema and annotations files
```

#### Step 3: Run the Notebook
1. Open `financial_document_vlm_finetuning.ipynb`
2. Run cells 1-10 to install dependencies and define classes
3. Uncomment and modify the main execution cell (Cell 12):

```python
# Define your data paths
PDF_FILES = [
    "./data/invoice1.pdf",
    "./data/invoice2.pdf",
    # Add more PDF files
]

SCHEMA_PATH = "./schema.json"
ANNOTATIONS_FILE = "./annotations.json"
OUTPUT_DIR = "./finetuned_qwen2vl"

# Run the complete pipeline
pipeline, eval_results = run_complete_pipeline(
    pdf_files=PDF_FILES,
    schema_path=SCHEMA_PATH,
    annotations_file=ANNOTATIONS_FILE,
    output_dir=OUTPUT_DIR
)
```

4. Run all cells

### 3. Configuration Options

Modify the `FineTuningConfig` class in Cell 2:

```python
@dataclass
class FineTuningConfig:
    # Model configuration
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    use_quantization: bool = True
    load_in_4bit: bool = True  # Use 4-bit for lower memory

    # LoRA configuration
    lora_r: int = 16  # Rank (higher = more parameters)
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.05

    # Training configuration
    num_train_epochs: int = 3  # Increase for better performance
    per_device_train_batch_size: int = 1  # Increase if you have enough VRAM
    gradient_accumulation_steps: int = 4  # Effective batch size = batch_size * this
    learning_rate: float = 2e-4

    # Image processing
    max_image_size: Tuple[int, int] = (1024, 1024)
    dpi: int = 200  # Higher DPI = better quality, more memory
```

## Memory Requirements

### GPU Memory
- **4-bit quantization**: ~8-10 GB VRAM
- **8-bit quantization**: ~14-16 GB VRAM
- **No quantization**: ~28+ GB VRAM

### System RAM
- Recommended: 40GB+ for processing PDFs and managing datasets

## Output Files

After training, you'll get:

```
./finetuned_qwen2vl/
  ├── adapter_config.json          # LoRA configuration
  ├── adapter_model.bin            # Fine-tuned adapter weights
  ├── training_config.json         # Training configuration
  ├── evaluation_results.json      # Evaluation metrics
  ├── preprocessor_config.json     # Processor configuration
  └── checkpoint-*/                # Training checkpoints
```

## Usage After Fine-tuning

### Load and Use the Fine-tuned Model

```python
from PIL import Image

# Load the fine-tuned model
pipeline = FineTuningPipeline(config)
pipeline.load_finetuned_model("./finetuned_qwen2vl")

# Create evaluator
evaluator = ModelEvaluator(
    pipeline.model,
    pipeline.processor,
    pipeline.tokenizer
)

# Make predictions
test_image = Image.open("invoice.png")
key_fields = ["invoice_number", "date", "total_amount", "vendor_name"]

result = evaluator.predict(test_image, key_fields)
print(result)
```

## Troubleshooting

### Out of Memory (OOM) Errors
1. Enable 4-bit quantization: `load_in_4bit=True`
2. Reduce batch size: `per_device_train_batch_size=1`
3. Reduce image size: `max_image_size=(768, 768)`
4. Reduce LoRA rank: `lora_r=8`

### Slow Training
1. Increase gradient accumulation: `gradient_accumulation_steps=8`
2. Enable mixed precision (automatic with FP16)
3. Use a more powerful GPU

### Poor Accuracy
1. Increase training epochs: `num_train_epochs=5`
2. Add more training data
3. Increase LoRA rank: `lora_r=32`
4. Adjust learning rate: `learning_rate=1e-4`

## Best Practices

### Data Preparation
- Ensure high-quality PDF scans (200+ DPI)
- Provide diverse training examples
- Validate all annotations carefully
- Include edge cases in training data

### Training
- Start with default hyperparameters
- Monitor validation loss to avoid overfitting
- Use TensorBoard for training visualization
- Save multiple checkpoints for comparison

### Evaluation
- Test on documents similar to production data
- Analyze per-field accuracy to identify weak areas
- Iterate on training data based on evaluation results

## File Structure

```
docu_parse_trainer/
├── financial_document_vlm_finetuning.ipynb  # Main notebook
├── README.md                                # This file
├── sample_schema.json                       # Example schema
├── sample_annotations.json                  # Example annotations
├── data/                                    # Your PDF files
├── processed_pdfs/                          # Generated images (auto-created)
└── finetuned_qwen2vl/                      # Output models (auto-created)
```

## Support

For issues or questions:
- Check the troubleshooting section
- Review Qwen2-VL documentation
- Check Runpod GPU compatibility

## License

This notebook is provided as-is for fine-tuning Qwen2-VL models. Please ensure you comply with:
- Qwen2-VL model license
- HuggingFace Transformers license
- Your organization's data usage policies

## Citation

If you use this notebook in your research or production, consider citing:
- Qwen2-VL: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
- LoRA: https://arxiv.org/abs/2106.09685
# Finetuning_vlms
