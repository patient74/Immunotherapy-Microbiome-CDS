"""
Configuration for the Microbiome-ICI Report Generator
"""

# =============================================================================
# Model Configuration
# =============================================================================

# MedGemma 1.5 4B model
MEDGEMMA_MODEL_ID = "google/medgemma-1.5-4b-it"
MEDGEMMA_DEVICE = "cuda"  # Change to "cpu" if no GPU available

# PubMedBERT embedding model
EMBEDDING_MODEL_ID = "pritamdeka/S-PubMedBert-MS-MARCO"
EMBEDDING_DEVICE = "cuda"

# =============================================================================
# Generation Parameters
# =============================================================================

# GENERATION_CONFIG = {
#     "temperature": 0.0,
#     "do_sample": False,
#     "repetition_penalty": 1.15,
#     "no_repeat_ngram_size": 5,
# }

GENERATION_CONFIG ={
  "temperature": 0.0,
  "do_sample": False,
  "top_p": 1.0,
  "top_k": 0,
  "repetition_penalty": 1.0,
  "num_beams": 1,
  "early_stopping": True,
}


SECTION_MAX_NEW_TOKENS = {
    "section_1": 1800,  
    "section_2": 1100,   
    "section_3": 3000,  
    "section_4": 1800,   
    "section_5": 2400,   
    "section_6": 1500,   
}

# =============================================================================
# ChromaDB Configuration
# =============================================================================

CHROMADB_COLLECTION_NAME = "research_papers"
CHROMADB_PERSIST_DIRECTORY = "/content/chroma_db"  # Adjust to your actual path

# =============================================================================
# RAG Retrieval Configuration
# =============================================================================

# Number of chunks to retrieve per section
RAG_TOP_K = {
    "section_1": 5,   # Composition Profile
    "section_2": 5,   # Metabolite Landscape
    "section_3": 9,   # Drug-Microbiome Interaction (most evidence-dense)
    "section_4": 7,   # Confounding Factors
    "section_5": 7,   # Intervention Considerations
}

# Metadata filtering strategy
# Options: "semantic_only", "metadata_only", "hybrid"
RETRIEVAL_STRATEGY = "hybrid"

# =============================================================================
# Report Configuration
# =============================================================================

OUTPUT_DIR = "./outputs"
REPORT_FILENAME_TEMPLATE = "microbiome_immunotherapy_report_{patient_id}_{timestamp}.md"

# =============================================================================
# Clinical Context Windows (days before therapy start)
# =============================================================================

ANTIBIOTIC_WINDOW_DAYS = 42  # Critical window for antibiotic impact (ICI)
ANTIBIOTIC_WINDOW_DAYS_ACT = 28  # Critical window before CAR-T infusion
PPI_CONCERN_DURATION_MONTHS = 3  # Duration after which PPI use is flagged

# ACT-specific toxicity windows
CRS_ONSET_DAYS = 14  # CRS typically occurs within 2 weeks of CAR-T infusion
NEUROTOXICITY_ONSET_DAYS = 21  # Neurotoxicity can occur up to 3 weeks post-infusion

# =============================================================================
# Taxa of Interest (for targeted retrieval)
# =============================================================================

KEY_TAXA = [
    "Akkermansia muciniphila",
    "Bifidobacterium",
    "Faecalibacterium prausnitzii",
    "Ruminococcaceae",
    "Lachnospiraceae",
    "Bacteroides",
    "Collinsella aerofaciens",
    "Alistipes",
    "Clostridium butyricum",
]

# =============================================================================
# Therapy Type Detection
# =============================================================================

THERAPY_TYPE_MAP = {
    # ICI drugs
    "pembrolizumab": "ICI",
    "nivolumab": "ICI",
    "atezolizumab": "ICI",
    "durvalumab": "ICI",
    "avelumab": "ICI",
    "ipilimumab": "ICI",
    "tremelimumab": "ICI",
    "cemiplimab": "ICI",
    
    # ACT drugs
    "tisagenlecleucel": "ACT",
    "axicabtagene ciloleucel": "ACT",
    "brexucabtagene autoleucel": "ACT",
    "lisocabtagene maraleucel": "ACT",
    "idecabtagene vicleucel": "ACT",
    "ciltacabtagene autoleucel": "ACT",
}

# =============================================================================
# ICI Drug Classes (for metadata filtering)
# =============================================================================

ICI_DRUG_CLASS_MAP = {
    "pembrolizumab": "PD-1/PD-L1 Blockade",
    "nivolumab": "PD-1/PD-L1 Blockade",
    "atezolizumab": "PD-1/PD-L1 Blockade",
    "durvalumab": "PD-1/PD-L1 Blockade",
    "avelumab": "PD-1/PD-L1 Blockade",
    "ipilimumab": "CTLA-4 Blockade",
    "tremelimumab": "CTLA-4 Blockade",
    "cemiplimab": "PD-1/PD-L1 Blockade",
}

# =============================================================================
# ACT Drug Classes (for metadata filtering)
# =============================================================================

ACT_DRUG_CLASS_MAP = {
    "tisagenlecleucel": "CAR-T (CD19-targeted)",
    "axicabtagene ciloleucel": "CAR-T (CD19-targeted)",
    "brexucabtagene autoleucel": "CAR-T (CD19-targeted)",
    "lisocabtagene maraleucel": "CAR-T (CD19-targeted)",
    "idecabtagene vicleucel": "CAR-T (BCMA-targeted)",
    "ciltacabtagene autoleucel": "CAR-T (BCMA-targeted)",
}

