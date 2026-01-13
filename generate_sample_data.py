#!/usr/bin/env python3
"""
Generate sample data for testing the financial document VLM fine-tuning pipeline.
This script creates sample schema and annotation files.
"""

import json
from pathlib import Path

def generate_sample_schema(output_path: str = "sample_schema.json"):
    """Generate a sample schema for financial documents"""
    schema = {
        "description": "Schema for extracting key fields from financial documents (invoices, receipts, etc.)",
        "version": "1.0",
        "fields": [
            {
                "name": "invoice_number",
                "type": "string",
                "required": True,
                "description": "Unique invoice or document identifier"
            },
            {
                "name": "date",
                "type": "string",
                "required": True,
                "description": "Invoice date (format: YYYY-MM-DD or as written)"
            },
            {
                "name": "due_date",
                "type": "string",
                "required": False,
                "description": "Payment due date"
            },
            {
                "name": "total_amount",
                "type": "number",
                "required": True,
                "description": "Total amount due including tax"
            },
            {
                "name": "subtotal",
                "type": "number",
                "required": False,
                "description": "Subtotal before tax"
            },
            {
                "name": "tax_amount",
                "type": "number",
                "required": False,
                "description": "Total tax amount"
            },
            {
                "name": "currency",
                "type": "string",
                "required": False,
                "description": "Currency code (e.g., USD, EUR, GBP)"
            },
            {
                "name": "vendor_name",
                "type": "string",
                "required": True,
                "description": "Name of the vendor/supplier"
            },
            {
                "name": "vendor_address",
                "type": "string",
                "required": False,
                "description": "Full vendor address"
            },
            {
                "name": "customer_name",
                "type": "string",
                "required": False,
                "description": "Name of the customer/buyer"
            },
            {
                "name": "payment_terms",
                "type": "string",
                "required": False,
                "description": "Payment terms (e.g., Net 30, Due on Receipt)"
            },
            {
                "name": "po_number",
                "type": "string",
                "required": False,
                "description": "Purchase order number"
            }
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2)

    print(f"✓ Sample schema created: {output_path}")
    return schema


def generate_sample_annotations(output_path: str = "sample_annotations.json"):
    """Generate sample annotations for testing"""
    annotations = [
        {
            "image_file": "processed_pdfs/invoice_001/page_1.png",
            "source": "training_dataset",
            "document_type": "invoice",
            "fields": {
                "invoice_number": "INV-2024-001",
                "date": "2024-01-15",
                "due_date": "2024-02-14",
                "total_amount": 1234.56,
                "subtotal": 1111.11,
                "tax_amount": 123.45,
                "currency": "USD",
                "vendor_name": "ACME Corporation",
                "vendor_address": "123 Business St, New York, NY 10001",
                "customer_name": "Tech Innovations Inc",
                "payment_terms": "Net 30",
                "po_number": "PO-2024-123"
            }
        },
        {
            "image_file": "processed_pdfs/invoice_002/page_1.png",
            "source": "training_dataset",
            "document_type": "invoice",
            "fields": {
                "invoice_number": "INV-2024-002",
                "date": "2024-01-16",
                "due_date": "2024-02-15",
                "total_amount": 5678.90,
                "subtotal": 5108.11,
                "tax_amount": 570.79,
                "currency": "USD",
                "vendor_name": "Tech Solutions Inc",
                "vendor_address": "456 Tech Ave, San Francisco, CA 94102",
                "customer_name": "Global Enterprises LLC",
                "payment_terms": "Net 30",
                "po_number": "PO-2024-456"
            }
        },
        {
            "image_file": "processed_pdfs/receipt_001/page_1.png",
            "source": "training_dataset",
            "document_type": "receipt",
            "fields": {
                "invoice_number": "RCP-2024-789",
                "date": "2024-01-17",
                "total_amount": 234.50,
                "subtotal": 215.14,
                "tax_amount": 19.36,
                "currency": "USD",
                "vendor_name": "Office Supplies Co",
                "customer_name": "Small Business LLC"
            }
        },
        {
            "image_file": "processed_pdfs/invoice_003/page_1.png",
            "source": "training_dataset",
            "document_type": "invoice",
            "fields": {
                "invoice_number": "2024-INV-1234",
                "date": "2024-01-20",
                "due_date": "2024-02-19",
                "total_amount": 9876.54,
                "subtotal": 8888.00,
                "tax_amount": 988.54,
                "currency": "EUR",
                "vendor_name": "European Services GmbH",
                "vendor_address": "Hauptstrasse 1, 10115 Berlin, Germany",
                "customer_name": "International Corp",
                "payment_terms": "Net 30",
                "po_number": "PO-EU-2024-789"
            }
        },
        {
            "image_file": "processed_pdfs/invoice_004/page_1.png",
            "source": "training_dataset",
            "document_type": "invoice",
            "fields": {
                "invoice_number": "INV/2024/00123",
                "date": "2024-01-25",
                "total_amount": 456.78,
                "tax_amount": 41.53,
                "currency": "GBP",
                "vendor_name": "UK Consultants Ltd",
                "customer_name": "London Business Group",
                "payment_terms": "Due on Receipt"
            }
        }
    ]

    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"✓ Sample annotations created: {output_path}")
    print(f"  Total samples: {len(annotations)}")
    return annotations


def generate_minimal_test_set(output_path: str = "minimal_test_annotations.json"):
    """Generate a minimal test set for quick testing"""
    annotations = [
        {
            "image_file": "test_data/sample_invoice.png",
            "source": "test_dataset",
            "document_type": "invoice",
            "fields": {
                "invoice_number": "TEST-001",
                "date": "2024-01-10",
                "total_amount": 100.00,
                "vendor_name": "Test Vendor Inc",
                "currency": "USD"
            }
        }
    ]

    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"✓ Minimal test set created: {output_path}")
    return annotations


def create_directory_structure():
    """Create the expected directory structure"""
    directories = [
        "data",
        "processed_pdfs",
        "finetuned_qwen2vl",
        "test_data"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def main():
    """Main function to generate all sample data"""
    print("=" * 60)
    print("Generating Sample Data for VLM Fine-tuning Pipeline")
    print("=" * 60)
    print()

    # Create directory structure
    print("Creating directory structure...")
    create_directory_structure()
    print()

    # Generate schema
    print("Generating sample schema...")
    generate_sample_schema()
    print()

    # Generate annotations
    print("Generating sample annotations...")
    generate_sample_annotations()
    print()

    # Generate minimal test set
    print("Generating minimal test set...")
    generate_minimal_test_set()
    print()

    print("=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print()
    print("Generated files:")
    print("  - sample_schema.json (field definitions)")
    print("  - sample_annotations.json (training examples)")
    print("  - minimal_test_annotations.json (quick test)")
    print()
    print("Next steps:")
    print("  1. Place your PDF files in the ./data/ directory")
    print("  2. Update the annotations with your actual data")
    print("  3. Run the Jupyter notebook to start training")
    print()
    print("Note: The sample annotations reference images that will be")
    print("      created when you process your PDFs through the pipeline.")


if __name__ == "__main__":
    main()
