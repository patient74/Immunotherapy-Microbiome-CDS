"""
Report assembler - combines sections into final markdown report
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

from . import config
from .section_generators import SectionGenerator

logger = logging.getLogger(__name__)


class ReportAssembler:
    """Assembles complete clinical report from individual sections.

    Supports two input modes:
    - JSON:  load_patient_data() / generate_and_save()           (existing path)
    - EHR:   load_patient_data_from_ehr() / generate_and_save_from_ehr()  (new path)
    """
    
    def __init__(self):
        self.generator = SectionGenerator()
    
    def load_patient_data(self, json_path: str) -> Dict:
        """Load patient JSON data from file"""
        logger.info(f"Loading patient data from {json_path}")
        
        with open(json_path, 'r') as f:
            patient_data = json.load(f)
        
        return patient_data

    def load_patient_data_from_ehr(self, ehr_path: str) -> Dict:
        """Extract patient JSON from a raw EHR text file using MedGemma.

        Args:
            ehr_path: Path to the plain-text EHR report.

        Returns:
            Patient data dictionary matching the pipeline schema.
        """
        # Imported here so the JSON-only code path has zero extra import cost
        from .ehr_extractor import EHRExtractor

        logger.info(f"Extracting patient data from EHR: {ehr_path}")
        extractor = EHRExtractor()
        return extractor.extract_from_file(ehr_path)
    
    def generate_full_report(self, patient_data: Dict) -> str:
        """
        Generate complete clinical report
        
        Args:
            patient_data: Patient JSON dictionary
            
        Returns:
            Complete report as markdown string
        """
        logger.info("Starting full report generation")
        
        report_sections = []
        
        # Section 0: Preamble (always included, not LLM-generated)
        logger.info("Generating preamble")
        preamble = self.generator.generate_preamble(patient_data)
        report_sections.append(preamble)
        
        # Section 1: Microbiome Composition Profile
        section_1 = self.generator.generate_section_1(patient_data)
        if section_1:
            report_sections.append(section_1)
        
        # Section 2: Metabolite Landscape
        section_2 = self.generator.generate_section_2(patient_data)
        if section_2:
            report_sections.append(section_2)
        
        # Section 3: Drug-Microbiome Interaction Outlook
        section_3 = self.generator.generate_section_3(patient_data)
        if section_3:
            report_sections.append(section_3)
        
        # Section 4: Confounding Factors
        section_4 = self.generator.generate_section_4(patient_data)
        if section_4:
            report_sections.append(section_4)
        
        # Section 5: Intervention Considerations
        section_5 = self.generator.generate_section_5(patient_data)
        if section_5:
            report_sections.append(section_5)
        
        # Section 6: Data Quality & Limitations (always included)
        section_6 = self.generator.generate_section_6(patient_data)
        report_sections.append(section_6)
        
        # References section
        references = self._generate_references_section()
        report_sections.append(references)
        
        # Footer
        footer = self._generate_footer()
        report_sections.append(footer)
        
        # Combine all sections
        full_report = "\n".join(report_sections)
        
        logger.info("Report generation complete")
        return full_report
    
    def _generate_references_section(self) -> str:
        """Generate references section from all citations and titles used"""
        # get_all_citations now returns List[tuple] i.e. [(citation, title), ...]
        references_data = self.generator.get_all_citations()
        
        if not references_data:
            return ""
        
        references = "## References\n\n"
        references += "The following peer-reviewed publications were cited in this report:\n\n"
        
        for i, (citation, title) in enumerate(references_data, 1):
            if title and title != citation:
                references += f"{i}. {citation}: {title}\n"
            else:
                references += f"{i}. {citation}\n"
        
        references += "\n"
        return references
    
    def _generate_footer(self) -> str:
        """Generate report footer with metadata"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        footer = f"""---

**Report Generated:** {timestamp}  
**Model:** MedGemma 1.5 4B  
**System:** Microbiome-Immunotherapy Clinical Decision Support v1.0

*This report is intended for use by qualified healthcare professionals as a clinical decision support tool. It does not constitute medical advice and should be interpreted in conjunction with comprehensive clinical evaluation.*
"""
        return footer
    
    def save_report(self, report: str, patient_id: str, output_dir: str = None) -> str:
        """
        Save report to markdown file
        
        Args:
            report: Complete report markdown string
            patient_id: Patient identifier for filename
            output_dir: Output directory (uses config default if not provided)
            
        Returns:
            Path to saved report file
        """
        if output_dir is None:
            output_dir = config.OUTPUT_DIR
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"microbiome_ici_report_{patient_id}_{timestamp}.md"
        filepath = output_path / filename
        
        # Save report
        with open(filepath, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to: {filepath}")
        return str(filepath)
    
    def generate_and_save(self, patient_json_path: str, output_dir: str = None) -> str:
        """
        Complete workflow: load data, generate report, save to file
        
        Args:
            patient_json_path: Path to patient JSON file
            output_dir: Optional output directory override
            
        Returns:
            Path to saved report file
        """
        # Load patient data
        patient_data = self.load_patient_data(patient_json_path)
        patient_id = patient_data["patient"]["id"]
        
        # Generate report
        report = self.generate_full_report(patient_data)
        
        # Save report
        output_path = self.save_report(report, patient_id, output_dir)
        
        return output_path

    def generate_and_save_from_ehr(
        self,
        ehr_path: str,
        output_dir: str = None,
        save_json_path: str = None,
    ) -> str:
        """
        Complete EHR workflow: extract JSON from EHR, generate report, save to file.

        Args:
            ehr_path: Path to the plain-text EHR report.
            output_dir: Optional output directory override.
            save_json_path: If provided, save the extracted patient JSON to this path
                            so it can be inspected or reused without re-running extraction.

        Returns:
            Path to the saved report markdown file.
        """
        # Step 1: Extract patient data from EHR
        patient_data = self.load_patient_data_from_ehr(ehr_path)
        patient_id = patient_data["patient"]["id"]

        # Step 2: Optionally save the extracted JSON
        if save_json_path:
            import json as _json
            from pathlib import Path as _Path
            _Path(save_json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_json_path, "w", encoding="utf-8") as f:
                _json.dump(patient_data, f, indent=2)
            logger.info(f"Extracted patient JSON saved to: {save_json_path}")

        # Step 3: Generate report
        report = self.generate_full_report(patient_data)

        # Step 4: Save report
        output_path = self.save_report(report, patient_id, output_dir)

        return output_path
