"""
Section-specific prompt templates for clinical report generation.
Optimized for instruction-following in smaller models (e.g. MedGemma 4B IT).

Key design principles applied:
- Single flat instruction block per section (no nested lists inside lists)
- Positive framing: tell the model what TO do, not what NOT to do
- Evidence anchor placed immediately before the generation task
- Citation format stated once, clearly, close to where citations are used
- Section headers kept inside the prompt so the model knows its structural role
- Global instruction kept minimal; section prompts are self-contained
"""

# =============================================================================
# Global Instruction (prepended to all section prompts)
# Kept short — section prompts carry the detailed guidance
# =============================================================================

GLOBAL_INSTRUCTION = """You are a clinical report writer assisting an oncologist.
Your output will become one section of a microbiome-immunotherapy report used to inform treatment decisions.

Two rules apply to every section you write:
- Every factual claim must come directly from the retrieved evidence provided. If the evidence does not address a topic, omit that topic.
- Every claim must be followed by an inline citation in this exact format: (Author et al., Journal Year). Only cite from the Retrieved evidence section below. The citations are present under "paper": "citation"

Write in formal clinical prose. Do not use bullet points unless explicitly instructed.
"""

# =============================================================================
# Section 1: Microbiome Diversity & Composition Profile
# =============================================================================

SECTION_1_PROMPT = """{global_instruction}

---
SECTION 1: Microbiome Diversity & Composition Profile
---

Patient context:
- Cancer type: {cancer_type} | Stage: {cancer_stage}
- Planned therapy: {drug_name} ({drug_class})

Patient microbiome data:
- Shannon Diversity Index: {shannon_index}
- Simpson Diversity Index: {simpson_index}
- Observed Species: {observed_species}
- Detected taxa (% relative abundance):
{detected_taxa}

Retrieved evidence:
{evidence}

Task:
Write this section in two parts.

Part 1 — Diversity characterization: Describe the patient's alpha diversity level. Use the retrieved evidence to characterize whether this diversity profile has been associated with favorable or unfavorable outcomes in this cancer and immunotherapy context. Cite the evidence.

Part 2 — Taxa characterization: For each detected taxon above that appears in the retrieved evidence, describe its observed relative abundance and what the evidence associates it with in this clinical context. Cover only taxa that have retrieved evidence. Cite each association.

Write in descriptive, factual prose. Do not predict this patient's individual outcome.

Begin writing Section 1 now:
"""

# =============================================================================
# Section 2: Metabolite Landscape
# =============================================================================

SECTION_2_PROMPT = """{global_instruction}

---
SECTION 2: Metabolite Landscape
---

Patient context:
- Cancer type: {cancer_type}
- Planned therapy: {drug_name} ({drug_class})

Patient metabolite data:
{metabolite_data}

Retrieved evidence:
{evidence}

Task:
Write a functional interpretation of the patient's metabolite profile. For each metabolite class present in the patient data (e.g. short-chain fatty acids, bile acids, tryptophan metabolites), do the following in sequence:
1. State the observed level from the patient data.
2. Describe what the retrieved evidence says about that metabolite class in the context of immune function and this therapy type. Cite the evidence.

If a metabolite class is present in the patient data but absent from the retrieved evidence, omit it entirely.
Frame this section as bridging microbiome composition to immune activity. Reserve response predictions for Section 3.

Begin writing Section 2 now:
"""

# =============================================================================
# Section 3: Drug–Microbiome Interaction Outlook (ICI version)
# =============================================================================

SECTION_3_ICI_PROMPT = """{global_instruction}

---
SECTION 3: Drug–Microbiome Interaction Outlook
---

Patient context:
- Cancer type: {cancer_type} | Stage: {cancer_stage}
- Planned therapy: {drug_name} ({drug_class}) | Line: {line_of_therapy}
- Tumor biomarkers: PD-L1 {pdl1} | TMB {tmb} | MSI {msi}

Patient microbiome summary:
- Shannon {shannon_index} | Simpson {simpson_index}
- Key taxa: {key_taxa_summary}
- Metabolite context: {metabolite_summary}

Retrieved evidence:
{evidence}

Task:
Write this section in three parts.

Part 1 — Overall microbiome-ICI context: Describe what the retrieved evidence says about how this patient's microbiome profile (diversity level and dominant taxa) compares to patterns observed in comparable cohorts treated with this ICI class. Use phrases such as "the evidence suggests" or "studies in comparable cohorts found". Cite all claims.

Part 2 — Individual taxa associations: For each taxon in the patient's key taxa list that appears in the retrieved evidence for this ICI class, describe the association the evidence reports (favorable, unfavorable, or bidirectional). If evidence reports both efficacy and immune-related adverse event (irAE) associations for the same taxon, state both explicitly. Cite each association.

Part 3 — Alpha diversity in this treatment setting: Describe what the retrieved evidence specifically says about alpha diversity and outcomes in this ICI and cancer type context. Cite the evidence.

Do not predict this individual patient's outcome. Attribute all findings to the evidence source.

Begin writing Section 3 now:
"""

# =============================================================================
# Section 3: Drug–Microbiome Interaction Outlook (ACT version)
# =============================================================================

SECTION_3_ACT_PROMPT = """{global_instruction}

---
SECTION 3: Microbiome–ACT Interaction Outlook
---

Patient context:
- Cancer type: {cancer_type} | Stage: {cancer_stage}
- Planned therapy: {drug_name} ({drug_class})
- ACT type: {act_type} | Target antigen: {target_antigen} | Cell source: {cell_source}
- Expected CRS risk: {crs_risk} | Expected neurotoxicity risk: {neurotoxicity_risk}
- Line of therapy: {line_of_therapy}

Patient microbiome summary:
- Shannon {shannon_index} | Simpson {simpson_index}
- Key taxa: {key_taxa_summary}
- Metabolite context: {metabolite_summary}

Retrieved evidence:
{evidence}

Task:
Write this section in four parts.

Part 1 — Overall microbiome-ACT context: Describe what the retrieved evidence says about how this patient's microbiome profile relates to outcomes observed in comparable ACT cohorts. Use phrases such as "the evidence suggests" or "studies in ACT cohorts found". Cite all claims.

Part 2 — Efficacy-related taxa: For each taxon in the patient's key taxa list where the retrieved evidence links it to CAR-T cell expansion, persistence, or anti-tumor cytotoxicity, describe that association. Cite each claim.

Part 3 — Toxicity-related taxa and metabolites: Describe what the retrieved evidence says about microbiota associations with CRS or ICANS risk. If the evidence links specific taxa or metabolites (particularly SCFAs) to T-cell function or inflammatory tone relevant to ACT toxicity, include those findings. Cite each claim.

Part 4 — Metabolite context for T-cell function: Describe what the retrieved evidence says about microbiota-derived metabolites, especially SCFAs, in modulating T-cell function in the ACT setting. Cite the evidence.

Do not predict this individual patient's outcome. Attribute all findings to the evidence source.

Begin writing Section 3 now:
"""

# =============================================================================
# Section 4: Confounding Factors
# =============================================================================

SECTION_4_PROMPT = """{global_instruction}

---
SECTION 4: Confounding Factors
---

Patient context:
- Cancer type: {cancer_type}
- Planned therapy: {drug_name} ({drug_class})

Patient confounding factor data:
{confounding_data}

Retrieved evidence:
{evidence}

Task:
For each confounding factor present in the patient data above, write one paragraph using only the retrieved evidence.

Antibiotic exposure: If present, describe what the retrieved evidence says about antibiotic timing relative to ICI initiation and its documented interactions with microbiome-mediated ICI efficacy. If the evidence distinguishes by antibiotic class, include that distinction. Cite the evidence.

PPI use: If present, describe what the retrieved evidence says about proton pump inhibitor effects on the microbiome in the ICI context. Cite the evidence.

Prior treatments: If prior chemotherapy or immunotherapy is recorded, describe any retrieved evidence connecting those treatments to microbiome changes relevant to subsequent ICI response. Cite the evidence.

Comorbidities: Include only if the retrieved evidence directly links the recorded comorbidity to microbiome-ICI interactions. Cite the evidence.

If no confounding factors are present in the patient data, or if no retrieved evidence addresses the recorded factors, output exactly this sentence:
"No significant confounding factors with established microbiome-immunotherapy interactions were identified in the available data."

Begin writing Section 4 now:
"""

# =============================================================================
# Section 5: Microbiota-Modulation Intervention Considerations
# =============================================================================

SECTION_5_PROMPT = """{global_instruction}

---
SECTION 5: Microbiota-Modulation Intervention Considerations
---

Patient context:
- Cancer type: {cancer_type}
- Planned therapy: {drug_name} ({drug_class})
- Microbiome context: {microbiome_summary}

Retrieved evidence by intervention type:
{evidence}

Task:
Generate a sub-section for each intervention type that has supporting evidence in the retrieved chunks above. Skip any intervention type with no retrieved evidence — do not note its absence.

For each sub-section that has evidence, use this structure:

Sub-section title (e.g., "Dietary and Prebiotic Approaches" or "Probiotic Supplementation")
Write 2–4 sentences describing what the retrieved evidence found about this intervention in this cancer and therapy context. State the finding, cite it, and note any caveat the evidence itself raises. Close with one sentence framing it as a consideration for clinical discussion rather than a recommendation.

Tone: exploratory and evidence-grounded. This section informs discussion; it does not prescribe.

Begin writing Section 5 now:
"""

# =============================================================================
# Section 6: Data Quality & Interpretive Limitations
# =============================================================================

SECTION_6_FIXED_CAVEATS = (
    "Microbiome composition is highly individual and dynamic; this report reflects a "
    "single time-point sample. Associations between microbiome features and immunotherapy response "
    "are derived from cohort-level studies and may not predict individual outcomes. "
    "The evidence base is evolving; findings should be interpreted in the context of "
    "current clinical judgment."
)

SECTION_6_PROMPT = """
---
SECTION 6: Data Quality & Interpretive Limitations
---

Patient sample data quality:
{data_quality}

Task:
Write 1–3 sentences addressing only what is specific to this patient's sample based on the data quality fields above.

Follow these rules in order:
- If completeness is "high" and no limitations are listed, write exactly: "No significant data quality limitations were identified for this sample."
- If completeness is below "high" or limitations are listed: name each affected data domain and state which section (Section 2 or Section 3) is consequently limited in its interpretation.
- If a metabolite class is listed as unavailable in the data quality fields, name it and describe the resulting interpretive gap. Only name classes that are explicitly listed as unavailable.

Do not add general statements about microbiome variability or evolving evidence — those appear in a separate fixed caveats section.

Begin writing Section 6 now:
"""


# =============================================================================
# Helper function to build full prompts
# =============================================================================

def build_prompt(section_name: str, patient_data: dict, evidence: str, **kwargs) -> str:
    """
    Build a complete prompt for a given section.

    Args:
        section_name: One of "section_1", "section_2", "section_3",
                      "section_4", "section_5", "section_6"
        patient_data: Patient JSON data dictionary
        evidence: Formatted evidence string from RAG
        **kwargs: Additional template variables (e.g. detected_taxa,
                  metabolite_data, confounding_data, data_quality, etc.)

    Returns:
        Complete formatted prompt string
    """
    therapy_type = patient_data["immunotherapy"].get("therapy_type", "ICI")

    # Select template
    if section_name == "section_3":
        template = SECTION_3_ACT_PROMPT if therapy_type == "ACT" else SECTION_3_ICI_PROMPT
    else:
        prompt_templates = {
            "section_1": SECTION_1_PROMPT,
            "section_2": SECTION_2_PROMPT,
            "section_4": SECTION_4_PROMPT,
            "section_5": SECTION_5_PROMPT,
            "section_6": SECTION_6_PROMPT,
        }
        template = prompt_templates.get(section_name)

    if not template:
        raise ValueError(f"Unknown section: {section_name}")

    # Inject global instruction (section_6 is standalone, doesn't use it)
    kwargs["global_instruction"] = GLOBAL_INSTRUCTION if section_name != "section_6" else ""
    kwargs["evidence"] = evidence

    # Inject shared patient context fields
    if section_name in ["section_1", "section_2", "section_3", "section_4", "section_5"]:
        kwargs["cancer_type"] = patient_data["cancer"]["type"]
        kwargs["drug_name"] = patient_data["immunotherapy"]["drug_name"]
        kwargs["drug_class"] = patient_data["immunotherapy"]["drug_class"]

    return template.format(**kwargs)
