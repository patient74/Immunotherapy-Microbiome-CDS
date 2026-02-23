"""
RAG retrieval logic with ChromaDB
"""

import json
import chromadb
from typing import List, Dict, Optional, Set
import logging

from . import config
from .models import get_embedding_model

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Handles retrieval from ChromaDB with metadata filtering"""
    
    def __init__(self):
        logger.info(f"Connecting to ChromaDB at {config.CHROMADB_PERSIST_DIRECTORY}")
        
        self.client = chromadb.PersistentClient(path=config.CHROMADB_PERSIST_DIRECTORY)
        self.collection = self.client.get_collection(name=config.CHROMADB_COLLECTION_NAME)
        self.embedding_model = get_embedding_model()
        
        logger.info(f"Connected to collection: {config.CHROMADB_COLLECTION_NAME}")
    
    def retrieve(
        self,
        query_text: str,
        top_k: int,
        metadata_filters: Optional[Dict] = None,
        exclude_filters: Optional[Dict] = None,
        strategy: str = "hybrid"  # options: "semantic_only", "metadata_only", "hybrid"
    ) -> List[Dict]:
        """
        Retrieve chunks from ChromaDB according to strategy:
        - semantic_only: ignores metadata filters, purely vector search
        - metadata_only: filters metadata, no fallback
        - hybrid: filters metadata, then fills remaining with semantic-only search
        """
        # Encode query
        query_embedding = self.embedding_model.encode_single(query_text)

        # Helper function to query Chroma
        def _query_chroma(where_clause=None, n_results=top_k):
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
            }
            if where_clause:
                query_params["where"] = where_clause
            results = self.collection.query(**query_params)
            chunks = []
            if results and results["documents"]:
                for i in range(len(results["documents"][0])):
                    chunk = {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else None,
                    }
                    chunks.append(chunk)
            return chunks

        # Build metadata filter clauses if needed
        include_clause = self._build_where_clause(metadata_filters)
        exclude_clause = self._build_exclude_clause(exclude_filters)
        if include_clause and exclude_clause:
            where_clause = {"$and": [include_clause, exclude_clause]}
        elif include_clause:
            where_clause = include_clause
        elif exclude_clause:
            where_clause = exclude_clause
        else:
            where_clause = None

        # --- Handle strategies ---
        if strategy == "semantic_only":
            return _query_chroma(where_clause=None, n_results=top_k)

        elif strategy == "metadata_only":
            # Only filter-based search, no fallback
            return _query_chroma(where_clause=where_clause, n_results=top_k)

        elif strategy == "hybrid":
            # Step 1: filtered search
            filtered_chunks = _query_chroma(where_clause=where_clause, n_results=top_k)

            # Step 2: fallback to semantic-only if not enough
            if len(filtered_chunks) < top_k:
                remaining_k = top_k - len(filtered_chunks)
                semantic_chunks = _query_chroma(where_clause=None, n_results=remaining_k)

                # Remove duplicates based on text
                existing_texts = set(c["text"] for c in filtered_chunks)
                semantic_chunks = [c for c in semantic_chunks if c["text"] not in existing_texts]

                filtered_chunks.extend(semantic_chunks)

            return filtered_chunks

        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _build_where_clause(self, filters: Optional[Dict]) -> Optional[Dict]:
        """
        Build a ChromaDB WHERE clause from an include-filter dictionary.

        Filters map tag category names (matching keys in research_papers.json "tags")
        to one or more values. The pipeline stores tags as flat pipe-delimited string
        fields named paper_tag_{category}, e.g.:
            paper_tag_cancer  = "NSCLC|Renal Cell Carcinoma|Bladder Cancer"
            paper_tag_treatment = "PD-1/PD-L1 Blockade"

        We use $contains for substring matching against the pipe-delimited string.

        Examples:
            {"cancer": "NSCLC"}
            -> {"paper_tag_cancer": {"$contains": "NSCLC"}}

            {"cancer": "NSCLC", "treatment": ["PD-1/PD-L1 Blockade"]}
            -> {"$and": [
                    {"paper_tag_cancer": {"$contains": "NSCLC"}},
                    {"paper_tag_treatment": {"$contains": "PD-1/PD-L1 Blockade"}},
               ]}

        Note: For list values, only the FIRST element is used for $contains filtering.
        If you need to match any of several values, call retrieve() once per value and
        merge results, or fetch unfiltered and post-filter in Python.
        """
        if not filters:
            return None
        
        where_conditions = []
        
        for key, value in filters.items():
            field = f"paper_tag_{key}"
            # Use the first element if a list was provided; $contains substring-matches
            # against the pipe-delimited string stored in the metadata field.
            match_value = value[0] if isinstance(value, list) else value
            where_conditions.append({field: {"$contains": match_value}})
        
        if len(where_conditions) == 1:
            return where_conditions[0]
        elif len(where_conditions) > 1:
            return {"$and": where_conditions}
        
        return None
    
    def _build_exclude_clause(self, filters: Optional[Dict]) -> Optional[Dict]:
        """
        Build a ChromaDB WHERE clause that EXCLUDES documents matching the filters.

        Uses $not_contains so chunks whose tag field contains the given value are
        filtered out.

        Example:
            {"section_type": "references"}
            -> {"section_type": {"$ne": "references"}}

            {"cancer": "NSCLC"}
            -> {"paper_tag_cancer": {"$not_contains": "NSCLC"}}
        """
        if not filters:
            return None
        
        where_conditions = []
        
        for key, value in filters.items():
            # Allow filtering on plain metadata fields (e.g. section_type)
            # as well as tag fields.
            if key.startswith("paper_tag_") or key in ("source_file", "section_type", "is_table"):
                field = key
            else:
                field = f"paper_tag_{key}"
            
            match_value = value[0] if isinstance(value, list) else value
            where_conditions.append({field: {"$not_contains": match_value}})
        
        if len(where_conditions) == 1:
            return where_conditions[0]
        elif len(where_conditions) > 1:
            return {"$and": where_conditions}
        
        return None

    def _parse_paper_meta(self, metadata: Dict) -> Dict:
        """
        Safely deserialise the paper metadata stored as a JSON string in ChromaDB.

        ChromaDB only stores scalar values, so the pipeline serialised the full
        paper dict as json.dumps(). This helper reverses that.
        """
        raw = metadata.get("paper", "{}")
        try:
            return json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            logger.warning("Could not deserialise paper metadata: %s", raw)
            return {}

    # ------------------------------------------------------------------
    # Section-specific retrieval methods
    # ------------------------------------------------------------------

    def retrieve_for_section_1(self, patient_data: Dict) -> List[Dict]:
        """
        Retrieve chunks for Section 1: Microbiome Composition Profile
        
        Focus on: diversity, detected taxa, cancer type, ICI class
        """
        cancer_type = patient_data["cancer"]["type"]
        ici_class = self._get_ici_class(patient_data["immunotherapy"]["drug_name"])
        
        detected_taxa = [
            taxon for taxon, abundance in patient_data["microbiome"]["key_bacteria"].items()
            if abundance is not None and abundance > 0
        ]
        
        query = f"""
        Microbiome composition and diversity in {cancer_type} patients receiving {ici_class} therapy.
        Taxa of interest: {', '.join(detected_taxa[:5])}.
        Alpha diversity and response to immunotherapy.
        """
        
        filters = {
            "cancer": cancer_type,
            "treatment": ici_class,
        }
        
        return self.retrieve(
            query_text=query,
            top_k=config.RAG_TOP_K["section_1"],
            metadata_filters=filters,
        )
    
    def retrieve_for_section_2(self, patient_data: Dict) -> List[Dict]:
        """
        Retrieve chunks for Section 2: Metabolite Landscape
        
        Focus on: SCFAs, bile acids, tryptophan metabolites
        """
        cancer_type = patient_data["cancer"]["type"]
        ici_class = self._get_ici_class(patient_data["immunotherapy"]["drug_name"])
        
        metabolites = patient_data["microbiome"]["metabolites"]
        
        metabolite_terms = []
        if metabolites["scfa"]["butyrate_uM"] is not None:
            metabolite_terms.append("short-chain fatty acids")
            metabolite_terms.append("butyrate")
        if metabolites["bile_acids_available"]:
            metabolite_terms.append("bile acids")
        if metabolites["tryptophan_metabolites_available"]:
            metabolite_terms.append("tryptophan metabolism")
        
        if not metabolite_terms:
            return []
        
        query = f"""
        Microbial metabolites and immune function in {cancer_type}.
        {', '.join(metabolite_terms)} and their role in immunotherapy response.
        CD8+ T cell function, regulatory T cells, mucosal immunity.
        """
        
        # Metabolite section: semantic search only (biology tags are too broad for
        # reliable filtering here).
        return self.retrieve(
            query_text=query,
            top_k=config.RAG_TOP_K["section_2"],
            metadata_filters=None,
            strategy="semantic_only"
        )
    
    def retrieve_for_section_3(self, patient_data: Dict) -> List[Dict]:
        """
        Retrieve chunks for Section 3: Drug-Microbiome Interaction Outlook
        """
        cancer_type = patient_data["cancer"]["type"]
        drug_name = patient_data["immunotherapy"]["drug_name"]
        therapy_type = self._get_therapy_type(patient_data)
        
        key_bacteria = patient_data["microbiome"]["key_bacteria"]
        detected_taxa = sorted(
            [(k, v) for k, v in key_bacteria.items() if v is not None and v > 0],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        taxa_names = [taxon for taxon, _ in detected_taxa]
        
        if therapy_type == "ICI":
            ici_class = self._get_ici_class(drug_name)
            query = f"""
            {ici_class} response prediction in {cancer_type} based on gut microbiome composition.
            Specific bacteria: {', '.join(taxa_names)}.
            Clinical outcomes, progression-free survival, response rates.
            Immune-related adverse events and microbiome associations.
            """
            filters = {"cancer": cancer_type, "treatment": ici_class}
        
        elif therapy_type == "ACT":
            act_details = patient_data["immunotherapy"].get("act_details", {})
            act_type = act_details.get("act_type", "CAR-T")
            target_antigen = act_details.get("target_antigen", "CD19")
            query = f"""
            {act_type} therapy efficacy in {cancer_type} and gut microbiome composition.
            Target antigen: {target_antigen}.
            Specific bacteria: {', '.join(taxa_names)}.
            CAR-T cell expansion, persistence, and anti-tumor activity.
            Cytokine release syndrome (CRS) and neurotoxicity associations with microbiome.
            T-cell function and microbiota-derived metabolites.
            """
            filters = {"cancer": cancer_type, "treatment": "CAR-T"}
        
        else:
            query = f"""
            Immunotherapy response in {cancer_type} and gut microbiome.
            Bacteria: {', '.join(taxa_names)}.
            """
            filters = {"cancer": cancer_type}
        
        return self.retrieve(
            query_text=query,
            top_k=config.RAG_TOP_K["section_3"],
            metadata_filters=filters
        )
    
    def retrieve_for_section_4(self, patient_data: Dict) -> List[Dict]:
        """
        Retrieve chunks for Section 4: Confounding Factors
        """
        cancer_type = patient_data["cancer"]["type"]
        therapy_type = self._get_therapy_type(patient_data)
        
        query_terms = []
        
        if patient_data["medications"]["antibiotic_history"]["recent_antibiotics"]:
            if therapy_type == "ACT":
                query_terms.append("antibiotic exposure before CAR-T therapy and outcomes")
                query_terms.append("gut microbiota disruption and CAR-T efficacy")
            else:
                query_terms.append("antibiotic exposure and immunotherapy outcomes")
            
        if patient_data["medications"]["ppi_use"]["currently_on_ppi"]:
            query_terms.append("proton pump inhibitors and microbiome")
        
        if patient_data["prior_treatments"]["chemotherapy"]["received"]:
            if therapy_type == "ACT":
                query_terms.append("prior chemotherapy effects on gut microbiota before CAR-T")
                query_terms.append("lymphodepleting chemotherapy and microbiome")
            else:
                query_terms.append("prior chemotherapy effects on gut microbiota")
        
        if not query_terms:
            return []
        
        query = f"""
        {' '.join(query_terms)} in {cancer_type} patients.
        Impact on immunotherapy efficacy and toxicity.
        """
        
        # Broader search for confounders — no cancer-type filter
        return self.retrieve(
            query_text=query,
            top_k=config.RAG_TOP_K["section_4"],
            metadata_filters=None,
            strategy="semantic_only"
        )
    
    def retrieve_for_section_5(self, patient_data: Dict) -> Dict[str, List[Dict]]:
        """
        Retrieve chunks for Section 5: Intervention Considerations
        """
        cancer_type = patient_data["cancer"]["type"]
        ici_class = self._get_ici_class(patient_data["immunotherapy"]["drug_name"])
        
        intervention_chunks = {}
        
        # Sub-section 5a: Dietary & Prebiotics
        diet_query = f"""
        Dietary interventions, prebiotics, fiber supplementation in {cancer_type}.
        High-fiber diet, inulin, pectin, polyphenols and immunotherapy response.
        """
        intervention_chunks["diet"] = self.retrieve(
            query_text=diet_query,
            top_k=10,
            metadata_filters=None,
            strategy="semantic_only"
        )
        
        # Sub-section 5b: Probiotics
        probiotics_query = f"""
        Probiotic supplementation in {cancer_type} patients receiving {ici_class}.
        Lactobacillus, Bifidobacterium, Akkermansia, Clostridium butyricum.
        Clinical trials and efficacy data.
        """
        intervention_chunks["probiotics"] = self.retrieve(
            query_text=probiotics_query,
            top_k=10,
            metadata_filters={"cancer": cancer_type}
        )
        
        return intervention_chunks
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_ici_class(self, drug_name: str) -> str:
        """Map drug name to ICI class"""
        return config.ICI_DRUG_CLASS_MAP.get(drug_name.lower(), "Immune Checkpoint Inhibitor")
    
    def _get_act_class(self, drug_name: str) -> str:
        """Map drug name to ACT class"""
        return config.ACT_DRUG_CLASS_MAP.get(drug_name.lower(), "Adoptive Cell Therapy")
    
    def _get_therapy_type(self, patient_data: Dict) -> str:
        """Determine therapy type from patient data"""
        if "therapy_type" in patient_data["immunotherapy"]:
            return patient_data["immunotherapy"]["therapy_type"]
        drug_name = patient_data["immunotherapy"]["drug_name"].lower()
        return config.THERAPY_TYPE_MAP.get(drug_name, "ICI")
    
    def format_chunks_for_llm(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into a structured string for LLM context.
        Returns markdown-formatted evidence with citations.
        """
        if not chunks:
            return "No relevant evidence retrieved."
        
        formatted = "# Retrieved Evidence\n\n"
        
        for i, chunk in enumerate(chunks, 1):
            # paper is stored as a JSON string in ChromaDB — deserialise it first
            paper_meta = self._parse_paper_meta(chunk["metadata"])
            
            citation = paper_meta.get("citation", "Unknown source")
            text = chunk["text"]
            
            formatted += f"## Evidence {i}\n"
            formatted += f"**Citation:** {citation}\n"
            formatted += f"**Content:** {text}\n\n"
        
        return formatted
    
    def get_unique_citations(self, chunks: List[Dict]) -> Set[str]:
        """Extract unique citations from chunks for a references section."""
        citations = set()
        for chunk in chunks:
            # paper is stored as a JSON string in ChromaDB — deserialise it first
            paper_meta = self._parse_paper_meta(chunk["metadata"])
            citation = paper_meta.get("citation")
            if citation:
                citations.add(citation)
        return citations

    def get_unique_citation_metadata(self, chunks: List[Dict]) -> Set[tuple]:
        """
        Extract unique (citation, title) tuples from chunks.
        Used for the final References section to show paper titles.
        """
        meta = set()
        for chunk in chunks:
            # paper is stored as a JSON string in ChromaDB — deserialise it first
            paper_meta = self._parse_paper_meta(chunk["metadata"])
            citation = paper_meta.get("citation")
            if citation:
                # Get title, falling back to citation if missing
                title = paper_meta.get("title", citation)
                meta.add((citation, title))
        return meta
