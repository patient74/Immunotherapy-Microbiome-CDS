# Medical RAG Pipeline for Research Papers

This directory contains the Retrieval-Augmented Generation (RAG) pipeline designed for extracting and processing medical research papers to support clinical decision-making in immunotherapy.

## Pipeline Overview

The pipeline transforms raw PDF research papers into a searchable vector database (ChromaDB), optimized for medical context and evidence retrieval.

- **PDF Extraction**: Uses `docling` for accurate markdown extraction from complex medical PDFs.
- **Cleaning**: `rag_md_cleaner.py` removes unnecessary metadata, references sections, and figures while preserving essential tables and text.
- **Chunking**: `SectionAwareChunker` implements section-aware splitting with specific handling for tables to ensure context is preserved.
- **Embedding**: Uses `pritamdeka/S-PubMedBert-MS-MARCO`, a domain-specific transformer model optimized for medical literature.
- **Storage**: Persists vectors and metadata in `ChromaDB`.

## Components

- `pdf_to_chromadb_pipeline.py`: The main entry point for the ingestion pipeline.
- `rag_md_cleaner.py`: Utility for cleaning extracted markdown.
- `research_papers.json`: Metadata registry (filename stem mapping to paper titles/citations).

## Data Requirements

To ensure accurate citations, provide a `research_papers.json` file in the same directory as the script. The format should be:

```json
{
 "1": {
        "reference_id": "25",
        "citation": "Takada et al., Int J Cancer 2021",
        "title": "Clinical impact of probiotics on the efficacy of anti-PD-1 monotherapy in patients with NSCLC",
        "year": 2021,
        "tags": {
            "treatment": [
                "PD-1/PD-L1 Blockade"
            ],
            "cancer": [
                "NSCLC"
            ],
            "biology": [
                "Gut microbiome composition",
                "Alpha diversity"
            ],
            "intervention": [
                "Probiotics"
            ]
        }
    },

```
*Where "1" is the filename stem of "1.pdf".*



## Usage

To run the pipeline and index a folder of PDFs:

```bash
python pdf_to_chromadb_pipeline.py --input-folder ./pdfs --db-path ./chroma_db
```


## Dependencies

Requires `docling`, `transformers`, `sentence-transformers`, and `chromadb`. See `requirements_pipeline.txt` for details.
