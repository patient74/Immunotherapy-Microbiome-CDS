"""
Section generation functions for each report section
"""

import logging
from typing import Dict, Optional, List

from .models import get_medgemma
from .rag import RAGRetriever
from .prompts import build_prompt
from . import config

logger = logging.getLogger(__name__)


class SectionGenerator:
    """Handles generation of individual report sections"""
    
    def __init__(self):
        self.llm = get_medgemma()
        self.rag = RAGRetriever()
        self.all_citations = {}  # Map citation -> title across all sections
    
    def generate_preamble(self, patient_data: Dict) -> str:
        """
        Generate Section 0: Clinical Preamble (auto-populated, no LLM)
        """
        p = patient_data["patient"]
        c = patient_data["cancer"]
        i = patient_data["immunotherapy"]
        m = patient_data["microbiome"]
        
        # Format metastases
        metastases_str = ", ".join(c["metastases"]) if c["metastases"] else "none"
        
        # Determine therapy type
        therapy_type = i.get("therapy_type", "ICI")
        
        preamble = f"""# Microbiome-Immunotherapy Clinical Report

**Patient ID:** {p['id']}  
**Age:** {p['age']} years  
**Gender:** {p['gender']}

## Clinical Context

**Cancer Diagnosis:** {c['stage']} {c['type']}"""
        
        if c.get('subtype'):
            preamble += f" ({c['subtype']})"
        
        preamble += f"""  
**Primary Site:** {c['primary_site']}  
**Metastases:** {metastases_str}  
**Diagnosis Date:** {c['diagnosis_date']}

**Tumor Biomarkers:**
- PD-L1 Expression: {c['biomarkers']['pdl1_expression']}
- Tumor Mutational Burden (TMB): {c['biomarkers']['tmb']}
- Microsatellite Instability (MSI): {c['biomarkers']['msi_status']}

## Planned Immunotherapy

**Therapy Type:** {therapy_type}  
**Drug:** {i['drug_name']} ({i['drug_class']})  
**Treatment Setting:** {i['treatment_setting']}  
**Line of Therapy:** {i['line_of_therapy']}  
**Planned Start Date:** {i['planned_start_date']}
"""
        
        # Add ACT-specific details if present
        if therapy_type == "ACT" and i.get("act_details"):
            act = i["act_details"]
            preamble += f"""
**ACT Details:**
- ACT Type: {act.get('act_type', 'N/A')}
- Target Antigen: {act.get('target_antigen', 'N/A')}
- Cell Source: {act.get('cell_source', 'N/A')}
- Preconditioning Regimen: {act.get('preconditioning_regimen', 'N/A')}
- T-Cell Harvest Date: {act.get('t_cell_harvest_date', 'N/A')}
- Expected CRS Risk: {act.get('expected_crs_risk', 'N/A')}
- Expected Neurotoxicity Risk: {act.get('expected_neurotoxicity_risk', 'N/A')}
"""
        
        preamble += f"""
## Microbiome Profile Overview

**Sample Date:** {m['sample_date']}  
**Sequencing Method:** {m['sequencing_method']}

This report summarizes gut microbiome findings relevant to anticipated immunotherapy response based on current evidence from peer-reviewed literature.

---
"""
        return preamble
    
    def generate_section_1(self, patient_data: Dict) -> Optional[str]:
        """
        Generate Section 1: Microbiome Diversity & Composition Profile
        """
        logger.info("Generating Section 1: Microbiome Diversity & Composition Profile")
        
        # Retrieve evidence
        chunks = self.rag.retrieve_for_section_1(patient_data)
        
        if not chunks:
            logger.warning("No evidence retrieved for Section 1, omitting section")
            return None
        
        # Track citations
        for citation, title in self.rag.get_unique_citation_metadata(chunks):
            self.all_citations[citation] = title
        
        # Format evidence
        evidence = self.rag.format_chunks_for_llm(chunks)
        
        # Prepare detected taxa string
        key_bacteria = patient_data["microbiome"]["key_bacteria"]
        detected_taxa_lines = []
        for taxon, abundance in key_bacteria.items():
            if abundance is not None and abundance > 0:
                taxon_display = taxon.replace("_", " ").title()
                detected_taxa_lines.append(f"- {taxon_display}: {abundance}%")
        
        detected_taxa_str = "\n".join(detected_taxa_lines) if detected_taxa_lines else "None detected above threshold"
        
        # Build prompt
        diversity = patient_data["microbiome"]["diversity"]
        prompt = build_prompt(
            "section_1",
            patient_data,
            evidence,
            cancer_stage=patient_data["cancer"]["stage"],
            shannon_index=diversity["shannon_index"],
            simpson_index=diversity["simpson_index"],
            observed_species=diversity["observed_species"],
            detected_taxa=detected_taxa_str,
        )
        
        # Generate
        content = self.llm.generate(prompt, max_new_tokens=config.SECTION_MAX_NEW_TOKENS["section_1"])
        
        return f"## 1. Microbiome Diversity & Composition Profile\n\n{content}\n\n"
    
    def generate_section_2(self, patient_data: Dict) -> Optional[str]:
        """
        Generate Section 2: Metabolite Landscape
        """
        logger.info("Generating Section 2: Metabolite Landscape")
        
        # Check if metabolite data is available
        metabolites = patient_data["microbiome"]["metabolites"]
        has_scfa = any(v is not None for v in metabolites["scfa"].values())
        has_metabolites = has_scfa or metabolites["bile_acids_available"] or metabolites["tryptophan_metabolites_available"]
        
        if not has_metabolites:
            logger.info("No metabolite data available, omitting Section 2")
            return None
        
        # Retrieve evidence
        chunks = self.rag.retrieve_for_section_2(patient_data)
        
        if not chunks:
            logger.warning("No evidence retrieved for Section 2, omitting section")
            return None
        
        # Track citations
        for citation, title in self.rag.get_unique_citation_metadata(chunks):
            self.all_citations[citation] = title
        
        # Format evidence
        evidence = self.rag.format_chunks_for_llm(chunks)
        
        # Prepare metabolite data string
        metabolite_lines = []
        
        if has_scfa:
            metabolite_lines.append("**Short-Chain Fatty Acids:**")
            scfa = metabolites["scfa"]
            if scfa["butyrate_uM"] is not None:
                metabolite_lines.append(f"- Butyrate: {scfa['butyrate_uM']} μM")
            if scfa["propionate_uM"] is not None:
                metabolite_lines.append(f"- Propionate: {scfa['propionate_uM']} μM")
            if scfa["acetate_uM"] is not None:
                metabolite_lines.append(f"- Acetate: {scfa['acetate_uM']} μM")
        
        if metabolites["bile_acids_available"]:
            metabolite_lines.append("**Bile Acids:** Analysis available")
        
        if metabolites["tryptophan_metabolites_available"]:
            metabolite_lines.append("**Tryptophan Metabolites:** Analysis available")
        
        metabolite_data_str = "\n".join(metabolite_lines)
        
        # Build prompt
        prompt = build_prompt(
            "section_2",
            patient_data,
            evidence,
            metabolite_data=metabolite_data_str,
        )
        
        # Generate
        content = self.llm.generate(prompt, max_new_tokens=config.SECTION_MAX_NEW_TOKENS["section_2"])
        
        return f"## 2. Metabolite Landscape\n\n{content}\n\n"
    
    def generate_section_3(self, patient_data: Dict) -> Optional[str]:
        """
        Generate Section 3: Drug–Microbiome Interaction Outlook
        Supports both ICI and ACT therapies
        """
        logger.info("Generating Section 3: Drug–Microbiome Interaction Outlook")
        
        # Retrieve evidence
        chunks = self.rag.retrieve_for_section_3(patient_data)
        
        if not chunks:
            logger.warning("No evidence retrieved for Section 3, omitting section")
            return None
        
        # Track citations
        for citation, title in self.rag.get_unique_citation_metadata(chunks):
            self.all_citations[citation] = title
        
        # Format evidence
        evidence = self.rag.format_chunks_for_llm(chunks)
        
        # Prepare summaries
        diversity = patient_data["microbiome"]["diversity"]
        key_bacteria = patient_data["microbiome"]["key_bacteria"]
        
        # Key taxa summary
        detected_taxa = [
            (k.replace("_", " ").title(), v)
            for k, v in key_bacteria.items()
            if v is not None and v > 0
        ]
        detected_taxa.sort(key=lambda x: x[1], reverse=True)
        key_taxa_summary = ", ".join([f"{t} ({a}%)" for t, a in detected_taxa[:5]])
        
        # Metabolite summary
        metabolites = patient_data["microbiome"]["metabolites"]
        metabolite_flags = []
        if any(v is not None for v in metabolites["scfa"].values()):
            metabolite_flags.append("SCFAs measured")
        if metabolites["bile_acids_available"]:
            metabolite_flags.append("bile acids available")
        if metabolites["tryptophan_metabolites_available"]:
            metabolite_flags.append("tryptophan metabolites available")
        metabolite_summary = ", ".join(metabolite_flags) if metabolite_flags else "limited metabolite data"
        
        # Determine therapy type
        therapy_type = patient_data["immunotherapy"].get("therapy_type", "ICI")
        
        # Build prompt based on therapy type
        if therapy_type == "ACT":
            act_details = patient_data["immunotherapy"].get("act_details", {})
            prompt = build_prompt(
                "section_3",
                patient_data,
                evidence,
                cancer_stage=patient_data["cancer"]["stage"],
                act_type=act_details.get("act_type", "CAR-T"),
                target_antigen=act_details.get("target_antigen", "CD19"),
                cell_source=act_details.get("cell_source", "autologous"),
                crs_risk=act_details.get("expected_crs_risk", "unknown"),
                neurotoxicity_risk=act_details.get("expected_neurotoxicity_risk", "unknown"),
                line_of_therapy=patient_data["immunotherapy"]["line_of_therapy"],
                shannon_index=diversity["shannon_index"],
                simpson_index=diversity["simpson_index"],
                key_taxa_summary=key_taxa_summary,
                metabolite_summary=metabolite_summary,
            )
        else:  # ICI
            biomarkers = patient_data["cancer"]["biomarkers"]
            prompt = build_prompt(
                "section_3",
                patient_data,
                evidence,
                cancer_stage=patient_data["cancer"]["stage"],
                pdl1=biomarkers["pdl1_expression"],
                tmb=biomarkers["tmb"],
                msi=biomarkers["msi_status"],
                line_of_therapy=patient_data["immunotherapy"]["line_of_therapy"],
                shannon_index=diversity["shannon_index"],
                simpson_index=diversity["simpson_index"],
                key_taxa_summary=key_taxa_summary,
                metabolite_summary=metabolite_summary,
            )
        
        # Generate
        content = self.llm.generate(prompt, max_new_tokens=config.SECTION_MAX_NEW_TOKENS["section_3"])  # Longer for this section
        
        # Section title varies by therapy type
        if therapy_type == "ACT":
            section_title = "3. Microbiome–ACT Interaction Outlook"
        else:
            section_title = "3. Drug–Microbiome Interaction Outlook"
        
        return f"## {section_title}\n\n{content}\n\n"
    
    def generate_section_4(self, patient_data: Dict) -> Optional[str]:
        """
        Generate Section 4: Confounding Factors
        """
        logger.info("Generating Section 4: Confounding Factors")
        
        # Check if any confounding factors are present
        meds = patient_data["medications"]
        prior = patient_data["prior_treatments"]
        
        has_confounders = (
            meds["antibiotic_history"]["recent_antibiotics"] or
            meds["ppi_use"]["currently_on_ppi"] or
            prior["chemotherapy"]["received"] or
            prior["prior_immunotherapy"]["received"] or
            len(patient_data.get("comorbidities", [])) > 0
        )
        
        if not has_confounders:
            logger.info("No confounding factors present, omitting Section 4")
            return None
        
        # Retrieve evidence
        chunks = self.rag.retrieve_for_section_4(patient_data)
        
        if not chunks:
            logger.warning("No evidence retrieved for Section 4, omitting section")
            return None
        
        # Track citations
        for citation, title in self.rag.get_unique_citation_metadata(chunks):
            self.all_citations[citation] = title
        
        # Format evidence
        evidence = self.rag.format_chunks_for_llm(chunks)
        
        # Prepare confounding data string
        confounding_lines = []
        
        # Antibiotic history
        if meds["antibiotic_history"]["recent_antibiotics"]:
            confounding_lines.append("**Antibiotic Exposure:**")
            for exp in meds["antibiotic_history"]["exposures"]:
                confounding_lines.append(
                    f"- {exp['antibiotic_name']} ({exp['antibiotic_class']}): "
                    f"{exp['start_date']} to {exp['end_date']} "
                    f"({exp['days_before_ici']} days before ICI start)"
                )
        
        # PPI use
        if meds["ppi_use"]["currently_on_ppi"]:
            ppi = meds["ppi_use"]
            confounding_lines.append(f"**Proton Pump Inhibitor Use:**")
            confounding_lines.append(f"- {ppi['ppi_name']}, duration: {ppi['duration_months']} months")
        
        # Prior chemotherapy
        if prior["chemotherapy"]["received"]:
            chemo = prior["chemotherapy"]
            regimens_str = ", ".join(chemo["regimens"])
            confounding_lines.append(f"**Prior Chemotherapy:**")
            confounding_lines.append(f"- Regimens: {regimens_str}")
            confounding_lines.append(f"- Response: {chemo['response']}")
        
        # Prior immunotherapy
        if prior["prior_immunotherapy"]["received"]:
            prior_ici = prior["prior_immunotherapy"]
            drugs_str = ", ".join(prior_ici["drugs"])
            confounding_lines.append(f"**Prior Immunotherapy:**")
            confounding_lines.append(f"- Drugs: {drugs_str}")
            confounding_lines.append(f"- Response: {prior_ici['response']}")
        
        # Comorbidities
        if patient_data.get("comorbidities"):
            comorbidities_str = ", ".join(patient_data["comorbidities"])
            confounding_lines.append(f"**Comorbidities:** {comorbidities_str}")
        
        confounding_data_str = "\n".join(confounding_lines)
        
        # Build prompt
        prompt = build_prompt(
            "section_4",
            patient_data,
            evidence,
            confounding_data=confounding_data_str,
        )
        
        # Generate
        content = self.llm.generate(prompt, max_new_tokens=config.SECTION_MAX_NEW_TOKENS["section_4"])
        
        return f"## 4. Confounding Factors\n\n{content}\n\n"
    
    def generate_section_5(self, patient_data: Dict) -> Optional[str]:
        """
        Generate Section 5: Microbiota-Modulation Intervention Considerations
        """
        logger.info("Generating Section 5: Microbiota-Modulation Intervention Considerations")
        
        # Retrieve evidence for each intervention type
        intervention_chunks = self.rag.retrieve_for_section_5(patient_data)
        
        # Check if any intervention evidence was retrieved
        total_chunks = sum(len(chunks) for chunks in intervention_chunks.values())
        if total_chunks == 0:
            logger.warning("No intervention evidence retrieved for Section 5, omitting section")
            return None
        
        # Track citations from all intervention types
        for chunks in intervention_chunks.values():
            for citation, title in self.rag.get_unique_citation_metadata(chunks):
                self.all_citations[citation] = title
        
        # Format evidence for each intervention type
        evidence_str = "## Diet and Prebiotics Evidence\n\n"
        evidence_str += self.rag.format_chunks_for_llm(intervention_chunks.get("diet", []))
        evidence_str += "\n\n## Probiotics Evidence\n\n"
        evidence_str += self.rag.format_chunks_for_llm(intervention_chunks.get("probiotics", []))
        
        # Prepare microbiome summary
        key_bacteria = patient_data["microbiome"]["key_bacteria"]
        detected_taxa = [
            k.replace("_", " ").title()
            for k, v in key_bacteria.items()
            if v is not None and v > 0
        ]
        microbiome_summary = f"Detected taxa: {', '.join(detected_taxa[:5])}"
        
        # Build prompt
        prompt = build_prompt(
            "section_5",
            patient_data,
            evidence_str,
            microbiome_summary=microbiome_summary,
        )
        
        # Generate
        content = self.llm.generate(prompt, max_new_tokens=config.SECTION_MAX_NEW_TOKENS["section_5"])
        
        return f"## 5. Microbiota-Modulation Intervention Considerations\n\n{content}\n\n"
    
    def generate_section_6(self, patient_data: Dict) -> str:
        """
        Generate Section 6: Data Quality & Interpretive Limitations
        """
        logger.info("Generating Section 6: Data Quality & Interpretive Limitations")
        
        # This section doesn't use RAG, just data quality fields
        data_quality = patient_data["microbiome"]["data_quality"]
        metabolites = patient_data["microbiome"]["metabolites"]
        
        # Prepare data quality string
        dq_lines = [
            f"**Sequencing Method:** {patient_data['microbiome']['sequencing_method']}",
            f"**Data Completeness:** {data_quality['completeness']}",
            f"**Data Source:** {data_quality['source']}",
        ]
        
        if data_quality.get("limitations"):
            dq_lines.append(f"**Limitations:** {', '.join(data_quality['limitations'])}")
        
        # Note missing metabolite data
        missing_metabolites = []
        if not any(v is not None for v in metabolites["scfa"].values()):
            missing_metabolites.append("Short-chain fatty acids")
        if not metabolites["bile_acids_available"]:
            missing_metabolites.append("Bile acids")
        if not metabolites["tryptophan_metabolites_available"]:
            missing_metabolites.append("Tryptophan metabolites")
        
        if missing_metabolites:
            dq_lines.append(f"**Missing Metabolite Data:** {', '.join(missing_metabolites)}")
        
        data_quality_str = "\n".join(dq_lines)

        from .prompts import SECTION_6_PROMPT, SECTION_6_FIXED_CAVEATS

        # Build prompt (no RAG evidence needed)
        prompt = SECTION_6_PROMPT.format(data_quality=data_quality_str)

        # Generate
        content = self.llm.generate(prompt, max_new_tokens=config.SECTION_MAX_NEW_TOKENS["section_6"])

        full_content = f"{SECTION_6_FIXED_CAVEATS}\n\n{content}"
        return f"## 6. Data Quality & Interpretive Limitations\n\n{full_content}\n\n"    
    def get_all_citations(self) -> List[tuple]:
        """Return sorted list of all unique (citation, title) tuples used in the report"""
        return sorted(list(self.all_citations.items()))
