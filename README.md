# Microbiome-Immunotherapy Clinical Decision Support System
### Evidence-based microbiome analytics to support cancer immunotherapy decisions

This project provides a sophisticated clinical decision support system that optimizes immunotherapy (ICI/ACT) treatments based on a patient's gut microbiome profile.

## Architecture Overview

The system processes patient data and clinical evidence through a modular pipeline to produce a 6-section clinical report:

1.  **Microbiome Composition**: Profile of diversity and key taxa.
2.  **Metabolite Landscape**: Analysis of SCFAs, bile acids, and tryptophan.
3.  **Drug-Microbiome Interaction**: Core interpretation of microbiome impact on drug efficacy.
4.  **Confounding Factors**: Impact of antibiotics, PPIs, and prior treatments.
5.  **Intervention Considerations**: Evidence-based dietary or probiotic suggestions.
6.  **Data Quality & Limitations**: Assessment of report confidence.

Each section is generated using targeted RAG retrieval from a database of peer-reviewed medical literature.

## Key Features

-   **EHR Extraction**: Automatically parses raw Electronic Health Records (EHR) text into structured patient data using MedGemma.
-   **Medical RAG**: Domain-specific retrieval system using PubMedBERT embeddings and table-aware chunking.
-   **Multi-Model Support**: Designed for MedGemma 1.5 but extensible to other LLMs.


## Project Structure

```text
├── rag/               # RAG pipeline for indexing medical papers
├── src/               # Core application logic (models, prompts, generators)
├── data/              # Patient data and clinical inputs
├── outputs/           # Generated clinical reports (Markdown)
├── generate_report.py # Main CLI entry point
└── requirements.txt   # Project dependencies
```

## Getting Started

### Prerequisites

-  First get the RAG ready. See `rag/README.md`
  (For demonstration purpose a ChromaDB database of around 40 papers is already hosted as a [HuggingFace Dataset](https://huggingface.co/datasets/fierce74/RAG-Immunotherapy-Microbiome-CDS) )
-   Python 3.10+
-   CUDA-compatible GPU (recommended for MedGemma and PubMedBERT)
-   HuggingFace access for `google/medgemma-1.5-4b-it`

### Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    pip install -r rag/requirements.txt
    ```

## Usage

Generate a report from structured patient JSON
(see data/template/patient_schema_template.json):
```bash
python generate_report.py data/patient_example.json
```

Generate a report from raw EHR text:
```bash
python generate_report.py data/patient_ehr.txt --save-ehr-json outputs/patient_profile.json
```
## Examples
See `data/sample_input` for EHR examples and `output` for the corresponding output  
## Configuration
See `src/config.py` to set the MAX_NEW_TOKENS and number of chunks to retrieve for each section of the report  

Settings for model IDs, device selection (CPU/GPU), and RAG parameters can be found in `src/config.py`.
