import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Dict

from .models import get_medgemma

logger = logging.getLogger(__name__)


# =============================================================================
# JSON Schema Template
# =============================================================================

# Helper to locate the template relative to this file
_BASE_DIR = Path(__file__).parent.parent
_SCHEMA_TEMPLATE_PATH = _BASE_DIR / "data" / "templates" / "patient_schema_template.json"

def _load_json_template() -> str:
    """Load the JSON schema template from the external file."""
    if not _SCHEMA_TEMPLATE_PATH.exists():
        logger.warning(f"Schema template not found at {_SCHEMA_TEMPLATE_PATH}. Extraction may fail or be inaccurate.")
        return "{}"
    
    with open(_SCHEMA_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _build_prompt(ehr_text: str) -> str:
    """
    Build the combined system + user prompt to pass to MedGemmaGenerator.generate().
    """
    today = date.today().isoformat()
    json_template = _load_json_template()

    system_instruction = f"""You are a clinical data extraction specialist for cancer immunotherapy. Extract structured data from EHRs covering both immune checkpoint inhibitors (ICI) and adoptive cell therapy (ACT, including CAR-T).

=== OUTPUT FORMAT ===
- Return ONLY the filled JSON object. No explanation, no preamble, no markdown fences.
- Do NOT add fields not in the template. Do NOT remove template fields.
- Valid JSON: no trailing commas, no comments, no extra keys.
- Set "extraction_date" to today: {today}.

=== DATA RULES ===
- Extract only explicitly stated values. Do not infer beyond specified rules.
- Dates: ISO 8601 (YYYY-MM-DD). If only month/year, use 1st (e.g. "March 2024" → "2024-03-01").
- Numbers: numeric type, not strings. Percentages as plain floats (4.8 not "4.8%").
- Missing optional fields: null.
- Missing required strings: "".
- Missing required arrays: [].
- Missing required booleans: false.

=== PATIENT ===
- "id": MRN exactly as written.

=== CANCER ===
- "type": Full name (e.g. "Diffuse Large B-Cell Lymphoma", "NSCLC", "Melanoma").
- "subtype": Histological subtype (e.g. "Non-GCB (ABC type)", "Adenocarcinoma").
- "stage": Use stage label only (e.g. "Stage IV", "IVA", "IIIB") — not full TNM.
- "metastases": List ANATOMICAL SITES with optional details in parentheses.
  Examples: ["Lung", "Liver"], ["Bone marrow (15% involvement)", "Pleural effusion (malignant)"].
  If M0 or no metastases, use [].
- "biomarkers.pdl1_expression": Use format from report. If percentage with TPS, use "<value>% TPS".
  If just percentage, use "<value>%". If N/A for non-relevant cancer types, use "N/A".
- "biomarkers.tmb": "<value> mutations/megabase" or "N/A" if not applicable.
- "biomarkers.msi_status": Full label (e.g. "MSS", "Microsatellite stable (MSS)") or "N/A".

=== IMMUNOTHERAPY (CRITICAL SECTION) ===
- "therapy_type": "ICI" for checkpoint inhibitors (pembrolizumab, nivolumab, ipilimumab, atezolizumab, durvalumab).
  "ACT" for adoptive cell therapy (CAR-T, TIL, TCR-T, etc.).
- "drug_name": Full drug name (e.g. "Pembrolizumab", "Axicabtagene ciloleucel").
- "drug_class": For ICI, use checkpoint target (e.g. "PD-1/PD-L1 Blockade", "PD-1 inhibitor").
  For ACT, use "CAR-T", "TIL therapy", "TCR-T", etc.
- "treatment_setting": "First-line", "Relapsed/Refractory", "consolidation", "metastatic", "adjuvant", "neoadjuvant".
- "line_of_therapy": "First-line", "Second-line", "Third-line", "consolidation", etc.
- "planned_start_date": Date therapy begins (for CAR-T, this is infusion date, not leukapheresis).

IF therapy_type is "ICI":
  - "ici_details": {{"ici_target": "PD-1", "PD-L1", "CTLA-4", or "PD-1, CTLA-4" for combinations}}
  - "act_details": null

IF therapy_type is "ACT":
  - "ici_details": null
  - "act_details": {{
      "act_type": "CAR-T", "TIL therapy", "TCR-T", etc.
      "target_antigen": e.g. "CD19", "CD22", "BCMA"
      "cell_source": "autologous" or "allogeneic"
      "preconditioning_regimen": e.g. "Fludarabine + Cyclophosphamide"
      "t_cell_harvest_date": Date of leukapheresis (YYYY-MM-DD)
      "expected_crs_risk": "low", "moderate", "moderate-high", "high"
      "expected_neurotoxicity_risk": "low", "moderate", "moderate-high", "high"
    }}

=== PRIOR TREATMENTS ===
- "chemotherapy.received": TRUE if any chemo regimen described (even if completed before current therapy).
- "chemotherapy.regimens": List each as string (e.g. ["R-CHOP", "R-ICE", "Gemcitabine (bridging)"]).
- "chemotherapy.response": Describe response to each regimen if stated.
- "prior_immunotherapy.received": TRUE only if immunotherapy given BEFORE current planned regimen.

=== MEDICATIONS ===
- "ppi_use.currently_on_ppi": true if on any PPI.
- "ppi_use.ppi_name": name (e.g. "Omeprazole").
- "ppi_use.duration_months": numeric months if stated; 0 if unknown.
- "antibiotic_history.recent_antibiotics": TRUE if any antibiotic within 90 days of planned therapy start.
- "antibiotic_history.exposures": List EVERY antibiotic course mentioned. Never leave [] if antibiotics documented.
  Each exposure object:
  - "antibiotic_name": name + dose (e.g. "Levofloxacin 500mg").
  - "antibiotic_class": use mappings (levofloxacin→fluoroquinolone, azithromycin→macrolide,
    piperacillin-tazobactam→beta-lactam, amoxicillin-clavulanate→beta-lactam/penicillin combination).
  - "start_date", "end_date": YYYY-MM-DD. If ongoing at report date, use "ongoing" for end_date.
  - "days_before_ici": Days from antibiotic END (or report date if ongoing) to planned therapy start.
  - "note" (optional): Add if context needed.

=== COMORBIDITIES ===
- List all conditions from Past Medical History as plain strings.
- Never use [] if a PMH section exists — scan fully.
- Include diet-controlled or asymptomatic conditions if listed.
- Do NOT include surgical history, family history, or social history.

=== MICROBIOME ===
- "sequencing_method": Exact method from report.
- "diversity.observed_species": Use "Observed OTUs" or "Observed Species" value.
- "key_bacteria": DYNAMIC object. Extract ALL bacterial species mentioned with abundance percentages.
  - Keys: lowercase, underscores for spaces (e.g. "akkermansia_muciniphila").
  - Create SEPARATE keys for each Bifidobacterium species — do NOT sum into bifidobacterium_spp unless explicitly stated.
  - Values: plain floats (percentages).
- "metabolites.scfa": butyrate, propionate, acetate as floats in μM. null if not measured.
- "metabolites.bile_acids_available": true if ANY bile acid data reported.
- "metabolites.tryptophan_metabolites_available": true if ANY tryptophan metabolite reported.
- "data_quality.completeness": "high" if metabolites + diversity + species all present; "moderate" if some missing; "low" if sparse.
- "data_quality.source": Lab name if stated; "" if unknown.
- "data_quality.limitations": Extract any explicitly noted limitations as strings.

=== CLINICAL CONTEXT ===
- "urgency": Extract urgency statements (e.g. "High - 33-day intervention window", "Standard"). "" if not stated.
- "patient_goals": List goals patient explicitly expressed.
- "specific_concerns": List clinical concerns from assessment.

=== FINAL CHECK ===
Before outputting, verify:
- therapy_type determines which of ici_details/act_details is populated (the other must be null).
- All antibiotic exposures logged in array.
- key_bacteria contains species-level entries from report (not a fixed schema).
- JSON is valid (no trailing commas, proper null usage)."""

    user_prompt = f"""EHR REPORT:
{ehr_text}

JSON TEMPLATE TO FILL:
{json_template}

Return the completed JSON object now."""

    return f"{system_instruction}\n\n{user_prompt}"


def _parse_output(raw_output: str) -> Dict:
    """
    Extract and parse a JSON object from raw model output.
    """

    # Strip any residual markdown fences (```json ... ``` or ``` ... ```)
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw_output)
    if fenced:
        raw_output = fenced.group(1)

    # Find the outermost JSON object
    start = raw_output.find("{")
    end = raw_output.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(
            "No JSON object found in model output.\n"
            f"Raw output (first 500 chars):\n{raw_output[:500]}"
        )

    json_str = raw_output[start:end]
    return json.loads(json_str)


class EHRExtractor:
    """
    Extracts structured patient JSON from free-text EHR reports using MedGemma.


    Usage:
        extractor = EHRExtractor()
        patient_data = extractor.extract(ehr_text)
        # or
        patient_data = extractor.extract_from_file("path/to/report.txt")
    """

    # # Expose as a static method so tests can call it without instantiation
    # _parse_output = staticmethod(_parse_output)

    def __init__(self):
        self._llm = get_medgemma()

#     def _get_llm(self):
#         if self._llm is None:
#             from .models import get_medgemma
#             self._llm = get_medgemma()
#         return self._llm

    def extract(self, ehr_text: str) -> Dict:
        """
        Run EHR extraction and return the parsed patient data dictionary.

        Args:
            ehr_text: Raw EHR report as a string.

        Returns:
            Patient data dict matching the pipeline's expected JSON schema.

        Raises:
            ValueError: If no valid JSON could be found in the model output.
            json.JSONDecodeError: If the extracted JSON string is malformed.
        """
        logger.info("Starting EHR extraction via MedGemma")

        prompt = _build_prompt(ehr_text)
        logger.info(f"EHR prompt length: {len(prompt)} characters")

        raw_output = self._llm.generate(prompt, max_new_tokens=6000)

        logger.debug(f"Raw EHR extraction output:\n{raw_output[:1000]}...")

        try:
            patient_data = _parse_output(raw_output)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error(
                f"EHR extraction failed — could not parse JSON.\n"
                f"Raw output:\n{raw_output}"
            )
            raise

        logger.info(
            f"EHR extraction complete. "
            f"Patient ID: {patient_data.get('patient', {}).get('id', 'unknown')}"
        )
        return patient_data

    def extract_from_file(self, ehr_path: str) -> Dict:
        """
        Load an EHR text file and extract patient data.

        Args:
            ehr_path: Path to the EHR text file.

        Returns:
            Patient data dict.
        """
        logger.info(f"Loading EHR from file: {ehr_path}")
        with open(ehr_path, "r", encoding="utf-8") as f:
            ehr_text = f.read()
        return self.extract(ehr_text)
