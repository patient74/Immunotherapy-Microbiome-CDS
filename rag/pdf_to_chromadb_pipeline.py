"""
pdf_to_chromadb_pipeline.py

Complete pipeline: PDF -> Docling extraction -> Cleaning -> Chunking -> ChromaDB
Optimized for research papers with PubMedBERT embeddings.

Pipeline steps:
1. Extract markdown from PDF using docling
2. Clean markdown (remove refs, metadata, figures; keep tables)
3. Chunk with section-awareness and table splitting
4. Embed using PubMedBERT tokenizer
5. Store in ChromaDB with metadata

Usage:
    python pdf_to_chromadb_pipeline.py --input-folder ./pdfs --db-path ./chroma_db
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import re
import unicodedata
from collections import defaultdict

# Import cleaning function
from rag_md_cleaner import clean_markdown_for_rag

# Docling imports
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("Warning: docling not installed. Install with: pip install docling")

# Transformers for tokenizer
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers torch")

# ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: chromadb not installed. Install with: pip install chromadb")

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")


# ========================================
# TABLE-AWARE MARKDOWN CHUNKER (embedded)
# ========================================

class TableAwareMarkdownSplitter:
    """Split markdown by headers while keeping tables intact."""
    
    def __init__(self, headers_to_split_on: List[tuple]):
        self.headers_to_split_on = sorted(
            headers_to_split_on, 
            key=lambda x: len(x[0]), 
            reverse=True
        )
    
    def split_text(self, text: str) -> List[Dict]:
        """Split text by headers while preserving table structure."""
        lines = text.split('\n')
        documents = []
        current_content = []
        current_metadata = {}
        in_table = False
        table_buffer = []
        
        for i, line in enumerate(lines):
            is_table_row = self._is_table_row(line)
            header_info = self._parse_header(line)
            
            if header_info and not in_table:
                if current_content:
                    documents.append({
                        'content': '\n'.join(current_content),
                        'metadata': current_metadata.copy(),
                        'has_table': False
                    })
                    current_content = []
                
                level, header_text = header_info
                current_metadata[level] = header_text
                self._clear_lower_headers(current_metadata, level)
                current_content.append(line)
                
            elif is_table_row:
                if not in_table:
                    if current_content:
                        caption = self._get_table_caption(current_content)
                        if caption:
                            current_content = current_content[:-1]
                            if current_content:
                                documents.append({
                                    'content': '\n'.join(current_content),
                                    'metadata': current_metadata.copy(),
                                    'has_table': False
                                })
                            current_content = []
                            table_buffer = [caption]
                        else:
                            documents.append({
                                'content': '\n'.join(current_content),
                                'metadata': current_metadata.copy(),
                                'has_table': False
                            })
                            current_content = []
                            table_buffer = []
                    in_table = True
                
                table_buffer.append(line)
                
            elif in_table and not is_table_row:
                in_table = False
                if table_buffer:
                    documents.append({
                        'content': '\n'.join(table_buffer),
                        'metadata': current_metadata.copy(),
                        'has_table': True
                    })
                    table_buffer = []
                current_content.append(line)
                
            else:
                current_content.append(line)
        
        if table_buffer:
            documents.append({
                'content': '\n'.join(table_buffer),
                'metadata': current_metadata.copy(),
                'has_table': True
            })
        
        if current_content:
            documents.append({
                'content': '\n'.join(current_content),
                'metadata': current_metadata.copy(),
                'has_table': False
            })
        
        return documents
    
    def _is_table_row(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        return stripped.startswith('|') or ('|' in stripped and stripped.count('|') >= 2)
    
    def _get_table_caption(self, content_lines: List[str]) -> Optional[str]:
        if not content_lines:
            return None
        last_line = content_lines[-1].strip()
        if re.match(r'^Table \d+[\.:].+', last_line, re.IGNORECASE):
            return last_line
        return None
    
    def _parse_header(self, line: str) -> Optional[tuple]:
        line = line.strip()
        for header_marker, level_name in self.headers_to_split_on:
            if line.startswith(header_marker + ' '):
                header_text = line[len(header_marker):].strip()
                return level_name, header_text
        return None
    
    def _clear_lower_headers(self, metadata: Dict, current_level: str):
        levels_order = [h[1] for h in self.headers_to_split_on]
        try:
            current_idx = levels_order.index(current_level)
            for level in levels_order[current_idx + 1:]:
                metadata.pop(level, None)
        except ValueError:
            pass


class SectionAwareChunker:
    """Chunk markdown with section awareness, table handling, and token limits."""
    
    def __init__(
        self, 
        model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        max_tokens: int = 330,
        chunk_overlap_tokens: int = 30,
        split_tables: bool = True
    ):
        """
        Initialize the chunker.
        
        Args:
            model_name: HuggingFace model name for tokenizer
            max_tokens: Maximum tokens per chunk
            chunk_overlap_tokens: Overlap between chunks in tokens
            split_tables: If True, split large tables by rows
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required. Install: pip install transformers torch")
        
        print(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded successfully")
        
        self.max_tokens = max_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.split_tables = split_tables
        
        self.headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ]
        
        self.header_splitter = TableAwareMarkdownSplitter(self.headers_to_split_on)
        
    def count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer."""
        return len(self.tokenizer.encode(text, add_special_tokens=True))
    
    def split_table_by_rows(self, table_text: str, max_tokens: int) -> List[str]:
        """Split a table into smaller chunks by rows."""
        lines = table_text.split('\n')
        
        caption = None
        header_row = None
        separator_row = None
        data_rows = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            if re.match(r'^Table \d+[\.:].+', line, re.IGNORECASE):
                caption = line
            elif '|' in line:
                if header_row is None:
                    header_row = line
                elif separator_row is None and re.match(r'^\|[\s\-:|]+\|', line):
                    separator_row = line
                else:
                    data_rows.append(line)
        
        if not data_rows:
            return [table_text]
        
        # Build header template
        header_template = []
        if caption:
            header_template.append(caption)
        if header_row:
            header_template.append(header_row)
        if separator_row:
            header_template.append(separator_row)
        
        header_tokens = self.count_tokens('\n'.join(header_template)) if header_template else 0
        
        # Check if a single row exceeds the limit
        max_row_tokens = max(self.count_tokens(row) for row in data_rows)
        
        if header_tokens + max_row_tokens > max_tokens:
            # Even a single row is too large - need to split columns
            print(f"Table row exceeds limit ({max_row_tokens} tokens), splitting columns...")
            return self._split_table_by_columns(table_text, max_tokens)
        
        # Split by rows normally
        chunks = []
        current_chunk_rows = []
        
        for row in data_rows:
            row_tokens = self.count_tokens(row)
            current_tokens = self.count_tokens('\n'.join(current_chunk_rows)) if current_chunk_rows else 0
            
            if header_tokens + current_tokens + row_tokens <= max_tokens:
                current_chunk_rows.append(row)
            else:
                if current_chunk_rows:
                    chunk = '\n'.join(header_template + current_chunk_rows)
                    chunks.append(chunk)
                # Start new chunk with this row
                current_chunk_rows = [row]
        
        if current_chunk_rows:
            chunk = '\n'.join(header_template + current_chunk_rows)
            chunks.append(chunk)
        
        return chunks if chunks else [table_text]
    
    def _split_table_by_columns(self, table_text: str, max_tokens: int) -> List[str]:
        """
        Split a wide table by columns when rows are too long.
        Creates multiple narrower tables, each with first column + subset of other columns.
        """
        lines = table_text.split('\n')
        
        caption = None
        header_row = None
        separator_row = None
        data_rows = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if re.match(r'^Table \d+[\.:].+', line, re.IGNORECASE):
                caption = line
            elif '|' in line:
                if header_row is None:
                    header_row = line
                elif separator_row is None and re.match(r'^\|[\s\-:|]+\|', line):
                    separator_row = line
                else:
                    data_rows.append(line)
        
        if not header_row or not data_rows:
            # Can't split intelligently, just return as text chunks
            return self.split_by_tokens(table_text, max_tokens, 0)
        
        # Parse header columns
        header_cells = [c.strip() for c in header_row.split('|')[1:-1]]
        n_cols = len(header_cells)
        
        if n_cols <= 2:
            # Too few columns to split, fall back to text splitting
            return self.split_by_tokens(table_text, max_tokens, 0)
        
        # Parse all rows into cells
        parsed_rows = []
        for row in data_rows:
            cells = [c.strip() for c in row.split('|')[1:-1]]
            # Pad or trim to match header length
            while len(cells) < n_cols:
                cells.append('')
            cells = cells[:n_cols]
            parsed_rows.append(cells)
        
        # Strategy: Keep first column (usually ID/key), split remaining columns into groups
        chunks = []
        
        # Try to fit columns into chunks
        first_col_idx = 0
        col_groups = []
        current_group = [first_col_idx]  # Always include first column
        
        for col_idx in range(1, n_cols):
            # Test if adding this column fits
            test_group = current_group + [col_idx]
            test_chunk = self._build_table_chunk(
                caption, header_cells, parsed_rows, test_group
            )
            test_tokens = self.count_tokens(test_chunk)
            
            if test_tokens <= max_tokens:
                current_group.append(col_idx)
            else:
                # Current group is full, save it
                if len(current_group) > 1:  # Has more than just first column
                    col_groups.append(current_group)
                current_group = [first_col_idx, col_idx]
        
        # Add remaining group
        if len(current_group) > 1:
            col_groups.append(current_group)
        
        # Build chunks from column groups
        for group_idx, col_indices in enumerate(col_groups):
            chunk = self._build_table_chunk(caption, header_cells, parsed_rows, col_indices)
            
            # Add note about which columns
            if len(col_groups) > 1:
                col_names = [header_cells[i] for i in col_indices[1:]]  # Skip first col (ID)
                note = f"\n[Table part {group_idx + 1}/{len(col_groups)}: columns {', '.join(col_names)}]"
                chunk = chunk + note
            
            chunks.append(chunk)
        
        return chunks if chunks else [table_text]
    
    def _build_table_chunk(
        self, 
        caption: Optional[str], 
        header_cells: List[str], 
        data_rows: List[List[str]], 
        col_indices: List[int]
    ) -> str:
        """Build a table chunk with selected columns."""
        lines = []
        
        if caption:
            lines.append(caption)
        
        # Header row with selected columns
        selected_headers = [header_cells[i] for i in col_indices]
        header_line = '| ' + ' | '.join(selected_headers) + ' |'
        lines.append(header_line)
        
        # Separator row
        separator = '| ' + ' | '.join(['---'] * len(col_indices)) + ' |'
        lines.append(separator)
        
        # Data rows with selected columns
        for row in data_rows:
            selected_cells = [row[i] if i < len(row) else '' for i in col_indices]
            row_line = '| ' + ' | '.join(selected_cells) + ' |'
            lines.append(row_line)
        
        return '\n'.join(lines)
    
    def split_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> List[str]:
        """Split text into chunks by token count."""
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|\n\n+', text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_tokens = self.count_tokens(sentence)
            
            if sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                word_chunks = self._split_by_words(sentence, max_tokens)
                if len(word_chunks) > 1:
                    chunks.extend(word_chunks[:-1])
                    current_chunk = [word_chunks[-1]]
                    current_tokens = self.count_tokens(word_chunks[-1])
                else:
                    chunks.extend(word_chunks)
                continue
            
            potential_tokens = current_tokens + sentence_tokens
            
            if potential_tokens <= max_tokens:
                current_chunk.append(sentence)
                current_tokens = potential_tokens
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                if overlap_tokens > 0 and current_chunk:
                    overlap_chunk = []
                    overlap_count = 0
                    for sent in reversed(current_chunk):
                        sent_tokens = self.count_tokens(sent)
                        if overlap_count + sent_tokens <= overlap_tokens:
                            overlap_chunk.insert(0, sent)
                            overlap_count += sent_tokens
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_tokens = overlap_count
                else:
                    current_chunk = []
                    current_tokens = 0
                
                current_chunk.append(sentence)
                current_tokens = current_tokens + sentence_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_by_words(self, text: str, max_tokens: int) -> List[str]:
        """Split text by words when sentences are too long."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for word in words:
            word_tokens = self.count_tokens(word + ' ')
            
            if current_tokens + word_tokens <= max_tokens:
                current_chunk.append(word)
                current_tokens += word_tokens
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_tokens = word_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_markdown(self, markdown_text: str, source_file: str = "unknown") -> List[Dict]:
        """Chunk markdown with section awareness and table handling."""
        header_splits = self.header_splitter.split_text(markdown_text)
        
        final_chunks = []
        
        for doc in header_splits:
            section_metadata = self._extract_section_info(doc['metadata'])
            is_table = doc.get('has_table', False)
            token_count = self.count_tokens(doc['content'])
            
            if token_count <= self.max_tokens:
                final_chunks.append({
                    "text": doc['content'],
                    "metadata": {
                        **section_metadata,
                        "token_count": token_count,
                        "is_table": is_table,
                        "chunk_index": 0,
                        "total_chunks_in_section": 1,
                        "source_file": source_file
                    }
                })
            elif is_table and self.split_tables:
                table_chunks = self.split_table_by_rows(doc['content'], self.max_tokens)
                for idx, chunk_text in enumerate(table_chunks):
                    final_chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            **section_metadata,
                            "token_count": self.count_tokens(chunk_text),
                            "is_table": True,
                            "chunk_index": idx,
                            "total_chunks_in_section": len(table_chunks),
                            "source_file": source_file
                        }
                    })
            elif is_table:
                # Keep table intact even if exceeds limit
                final_chunks.append({
                    "text": doc['content'],
                    "metadata": {
                        **section_metadata,
                        "token_count": token_count,
                        "is_table": True,
                        "chunk_index": 0,
                        "total_chunks_in_section": 1,
                        "source_file": source_file,
                        "exceeds_limit": True
                    }
                })
            else:
                sub_chunks = self.split_by_tokens(
                    doc['content'], 
                    self.max_tokens,
                    self.chunk_overlap_tokens
                )
                
                for idx, chunk_text in enumerate(sub_chunks):
                    final_chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            **section_metadata,
                            "token_count": self.count_tokens(chunk_text),
                            "is_table": False,
                            "chunk_index": idx,
                            "total_chunks_in_section": len(sub_chunks),
                            "source_file": source_file
                        }
                    })
        
        return final_chunks
    
    def _extract_section_info(self, metadata: Dict) -> Dict:
        """Extract section information from metadata."""
        section_info = {}
        
        for level in ['h1', 'h2', 'h3', 'h4']:
            if level in metadata:
                section_info[level] = metadata[level]
        
        section_type = self._identify_section_type(metadata)
        if section_type:
            section_info['section_type'] = section_type
        
        return section_info
    
    def _identify_section_type(self, metadata: Dict) -> str:
        """Identify section type based on header text."""
        all_headers = ' '.join([
            metadata.get('h1', ''),
            metadata.get('h2', ''),
            metadata.get('h3', ''),
            metadata.get('h4', '')
        ]).lower()
        
        section_patterns = {
            'abstract': r'\babstract\b',
            'introduction': r'\bintroduction\b',
            'background': r'\bbackground\b',
            'literature_review': r'\bliterature review\b|\brelated work\b',
            'methodology': r'\bmethodology\b|\bmethods\b|\bmaterials and methods\b',
            'results': r'\bresults\b',
            'discussion': r'\bdiscussion\b',
            'conclusion': r'\bconclusion\b|\bconcluding remarks\b',
            'references': r'\breferences\b|\bbibliography\b',
            'appendix': r'\bappendix\b',
            'acknowledgments': r'\backnowledgments\b|\backnowledgements\b',
            'abbreviations': r'\babbreviations\b',
            'data_availability': r'\bdata availability\b'
        }
        
        for section_type, pattern in section_patterns.items():
            if re.search(pattern, all_headers):
                return section_type
        
        return 'other'


# ========================================
# PIPELINE CLASS
# ========================================

class PDFToChromaDBPipeline:
    """Complete pipeline from PDF to ChromaDB."""
    
    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "research_papers",
        embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        max_tokens: int = 330,
        chunk_overlap: int = 30,
        papers_json: Optional[str] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            db_path: Path to ChromaDB database
            collection_name: Name of ChromaDB collection
            embedding_model: HuggingFace model for embeddings
            max_tokens: Maximum tokens per chunk
            chunk_overlap: Overlap between chunks
            papers_json: Path to filename-keyed JSON with paper metadata.
                         Keys are PDF stems (no extension), e.g. {"1": {...}, "2": {...}}
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap

        # Load paper registry (filename stem -> paper info dict)
        self.paper_registry = self._load_paper_registry(papers_json)

        # Initialize components
        self._init_docling()
        self._init_chunker()
        self._init_embedder()
        self._init_chromadb()
        
    def _load_paper_registry(self, papers_json: Optional[str]) -> Dict:
        """Load the filename-keyed paper metadata JSON.
        
        Args:
            papers_json: Path to JSON file. Expected format:
                         { "<pdf_stem>": { <paper fields> }, ... }
        Returns:
            Dict mapping pdf stem -> paper info, or empty dict if not provided.
        """
        if not papers_json:
            print("ℹ No papers JSON provided — paper metadata will not be attached to chunks.")
            return {}
        
        try:
            with open(papers_json, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            print(f"✓ Loaded paper registry from {papers_json} ({len(registry)} entries)")
            return registry
        except FileNotFoundError:
            print(f"⚠ Papers JSON not found: {papers_json} — continuing without paper metadata.")
            return {}
        except json.JSONDecodeError as e:
            print(f"⚠ Failed to parse papers JSON: {e} — continuing without paper metadata.")
            return {}

    def _get_paper_info(self, pdf_stem: str) -> Dict:
        """Look up paper metadata by PDF filename stem.
        
        Args:
            pdf_stem: PDF filename without extension, e.g. '1' for '1.pdf'
        Returns:
            Paper info dict, or empty dict if not found.
        """
        info = self.paper_registry.get(pdf_stem, {})
        if not info:
            print(f"  ⚠ No paper metadata found for '{pdf_stem}' in registry.")
        return info

    def _init_docling(self):
        """Initialize docling converter."""
        if not DOCLING_AVAILABLE:
            raise ImportError("docling required. Install: pip install docling")
        self.converter = DocumentConverter()
        print("✓ Docling converter initialized")
        
    def _init_chunker(self):
        """Initialize chunker with tokenizer."""
        self.chunker = SectionAwareChunker(
            model_name=self.embedding_model_name,
            max_tokens=self.max_tokens,
            chunk_overlap_tokens=self.chunk_overlap,
            split_tables=True
        )
        print("✓ Chunker initialized")
        
    def _init_embedder(self):
        """Initialize embedding model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required. Install: pip install sentence-transformers")
        
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        print("✓ Embedding model loaded")
        
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb required. Install: pip install chromadb")
        
        # Create persistent client
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✓ ChromaDB initialized at {self.db_path}")
        print(f"✓ Collection '{self.collection_name}' ready (existing docs: {self.collection.count()})")
        
    def extract_pdf(self, pdf_path: str) -> str:
        """Extract markdown from PDF using docling."""
        print(f"  Extracting PDF: {pdf_path}")
        result = self.converter.convert(pdf_path)
        markdown_text = result.document.export_to_markdown()
        print(f"  ✓ Extracted {len(markdown_text)} characters")
        return markdown_text
        
    def clean_markdown(self, markdown_text: str) -> str:
        """Clean markdown using rag_md_cleaner."""
        print(f"  Cleaning markdown...")
        cleaned = clean_markdown_for_rag(
            markdown_text,
            remove_tables=False,  # Keep tables
            remove_figures=True,
            remove_references=True,
            reference_mode="conservative",
            remove_metadata=True,
        )
        print(f"  ✓ Cleaned to {len(cleaned)} characters")
        return cleaned
        
    def chunk_text(self, text: str, source_file: str) -> List[Dict]:
        """Chunk text with section awareness, attaching paper metadata."""
        print(f"  Chunking text...")
        chunks = self.chunker.chunk_markdown(text, source_file=source_file)

        # Attach paper metadata to every chunk
        paper_info = self._get_paper_info(source_file)
        for chunk in chunks:
            chunk['metadata']['paper'] = paper_info

        # Validation
        max_tokens = max(c['metadata']['token_count'] for c in chunks)
        table_chunks = sum(1 for c in chunks if c['metadata'].get('is_table'))

        print(f"  ✓ Created {len(chunks)} chunks")
#         print(f"    - Text chunks: {len(chunks) - table_chunks}")
#         print(f"    - Table chunks: {table_chunks}")
        print(f"    - Max tokens: {max_tokens}")
        if paper_info:
            print(f"    - Paper: {paper_info.get('title', paper_info.get('reference_id', '?'))}")

        return chunks
        
    def embed_chunks(self, chunks: List[Dict]) -> List[List[float]]:
        """Create embeddings for chunks."""
        print(f"  Embedding {len(chunks)} chunks...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        print(f"  ✓ Created embeddings")
        return embeddings.tolist()
        
    def store_in_chromadb(
        self, 
        chunks: List[Dict], 
        embeddings: List[List[float]],
        pdf_filename: str
    ):
        """Store chunks and embeddings in ChromaDB."""
        print(f"  Storing in ChromaDB...")
        
        # Prepare data
        ids = [f"{pdf_filename}_{i}" for i in range(len(chunks))]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            # Flatten metadata for ChromaDB (ChromaDB only accepts scalar values)
            metadata = {
                'source_file': chunk['metadata'].get('source_file', ''),
                'section_type': chunk['metadata'].get('section_type', 'other'),
                'is_table': str(chunk['metadata'].get('is_table', False)),
                'token_count': chunk['metadata']['token_count'],
                'chunk_index': chunk['metadata']['chunk_index'],
                'timestamp': datetime.now().isoformat(),
            }

            # Add header hierarchy
            for level in ['h1', 'h2', 'h3', 'h4']:
                if level in chunk['metadata']:
                    metadata[level] = chunk['metadata'][level]

            # Attach paper metadata — serialised as JSON string so ChromaDB accepts it.
            # To retrieve: json.loads(chunk_metadata['paper'])
            paper_info = chunk['metadata'].get('paper', {})
            metadata['paper'] = json.dumps(paper_info) if paper_info else '{}'

            # Also write each tag array as a flat pipe-delimited string field so the
            # retriever can filter with ChromaDB where clauses (which only support scalars).
            # e.g. paper_tag_cancer = "NSCLC|Renal Cell Carcinoma|Bladder Cancer"
            # Retriever filters with: {"paper_tag_cancer": {"$contains": "NSCLC"}}
            tags = paper_info.get('tags', {}) if paper_info else {}
            for tag_key, tag_values in tags.items():
                if isinstance(tag_values, list) and tag_values:
                    metadata[f'paper_tag_{tag_key}'] = '|'.join(str(v) for v in tag_values)

            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"  ✓ Stored {len(chunks)} chunks in ChromaDB")
        print(f"  ✓ Total documents in collection: {self.collection.count()}")
        
    def process_pdf(self, pdf_path: str) -> Dict:
        """Process a single PDF through the entire pipeline."""
        pdf_filename = Path(pdf_path).stem
        print(f"\n{'='*80}")
        print(f"Processing: {pdf_filename}")
        print(f"{'='*80}")
        
        try:
            # Extract
            markdown = self.extract_pdf(pdf_path)

            # Clean
            cleaned = self.clean_markdown(markdown)

            # Chunk (paper metadata is attached inside chunk_text)
            chunks = self.chunk_text(cleaned, source_file=pdf_filename)
            
#             with open("/content/files/chunks/" + pdf_filename + ".json", "w", encoding="utf-8") as f:
#                 json.dump(chunks, f, indent=2, ensure_ascii=False)
#             print(f"\n✓ Saved {len(chunks)} chunks")
            
            # Embed
            embeddings = self.embed_chunks(chunks)
            
            # Store
            self.store_in_chromadb(chunks, embeddings, pdf_filename)
            
            result = {
                'status': 'success',
                'pdf_file': pdf_filename,
                'num_chunks': len(chunks),
                'max_tokens': max(c['metadata']['token_count'] for c in chunks),
            }
            
            print(f"✓ Successfully processed {pdf_filename}")
            return result
            
        except Exception as e:
            print(f"✗ Error processing {pdf_filename}: {str(e)}")
            return {
                'status': 'error',
                'pdf_file': pdf_filename,
                'error': str(e)
            }
            
    def process_folder(self, input_folder: str) -> List[Dict]:
        """Process all PDFs in a folder."""
        pdf_files = list(Path(input_folder).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {input_folder}")
            return []
        
        print(f"\nFound {len(pdf_files)} PDF files to process")
        
        results = []
        for pdf_path in pdf_files:
            result = self.process_pdf(str(pdf_path))
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful
        
        print(f"\n{'='*80}")
        print("PIPELINE SUMMARY")
        print(f"{'='*80}")
        print(f"Total PDFs: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if successful > 0:
            total_chunks = sum(r.get('num_chunks', 0) for r in results if r['status'] == 'success')
            print(f"Total chunks created: {total_chunks}")
            print(f"ChromaDB collection size: {self.collection.count()}")
        
        return results
    
    def query(
        self, 
        query_text: str, 
        n_results: int = 5,
        filter_section: Optional[str] = None
    ) -> Dict:
        """Query the ChromaDB collection."""
        # Create query embedding
        query_embedding = self.embedding_model.encode(
            [query_text],
            normalize_embeddings=True
        )[0].tolist()
        
        # Build filter
        where = {}
        if filter_section:
            where['section_type'] = filter_section
        
        # Query
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where if where else None
        )
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='PDF to ChromaDB Pipeline with PubMedBERT'
    )
    parser.add_argument(
        '--input-folder',
        required=True,
        help='Folder containing PDF files'
    )
    parser.add_argument(
        '--db-path',
        default='./chroma_db',
        help='Path to ChromaDB database (default: ./chroma_db)'
    )
    parser.add_argument(
        '--collection-name',
        default='research_papers',
        help='ChromaDB collection name (default: research_papers)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=330,
        help='Maximum tokens per chunk (default: 330)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=30,
        help='Chunk overlap in tokens (default: 30)'
    )
    parser.add_argument(
        '--embedding-model',
        default='pritamdeka/S-PubMedBert-MS-MARCO',
        help='HuggingFace embedding model'
    )
    parser.add_argument(
        '--papers-json',
        default=None,
        help='Path to filename-keyed paper metadata JSON (e.g. research_papers.json)'
    )

    args = parser.parse_args()
    
    # Initialize pipeline
    print("\nInitializing PDF to ChromaDB Pipeline...")
    print(f"{'='*80}")
    
    pipeline = PDFToChromaDBPipeline(
        db_path=args.db_path,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        max_tokens=args.max_tokens,
        chunk_overlap=args.overlap,
        papers_json=args.papers_json,
    )
    
    # Process folder
    results = pipeline.process_folder(args.input_folder)
    
    # Save results log
    log_file = f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {log_file}")


if __name__ == "__main__":
    main()
