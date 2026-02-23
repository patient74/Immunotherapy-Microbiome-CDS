#!/usr/bin/env python3
"""
Main CLI entry point

Accepts either:
  - A pre-built patient JSON file (check template) (.json)
  - A raw EHR text file          

Examples:
  python generate_report.py patient_example.json
  python generate_report.py patient_ehr.txt
  python generate_report.py patient_ehr.txt --save-ehr-json extracted_patient.json
"""

import argparse
import logging
import sys
from pathlib import Path

from src.report_assembler import ReportAssembler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate microbiome-immunotherapy clinical decision support report. "
            "Accepts either a pre-built patient JSON file or a raw EHR text file."
        )
    )

    parser.add_argument(
        "patient_input",
        type=str,
        help=(
            "Path to patient data file. "
            "Use a .json file for pre-extracted patient data, "
            "or a .txt/.ehr file to extract from a raw EHR report first."
        )
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated report (default: ./outputs)"
    )

    parser.add_argument(
        "--save-ehr-json",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "[EHR mode only] Save the MedGemma-extracted patient JSON to this path. "
            "Useful for inspecting or reusing the extraction without re-running the model."
        )
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.patient_input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Detect input mode by extension
    is_json = input_path.suffix.lower() == ".json"

    if not is_json and args.save_ehr_json is None:
        logger.info(
            "Tip: use --save-ehr-json <path> to save the extracted patient JSON "
            "and skip re-extraction on future runs."
        )

    logger.info("=" * 80)
    logger.info("Microbiome-ICI Clinical Report Generator v1.0")
    logger.info("=" * 80)

    if is_json:
        logger.info(f"Input mode: pre-built patient JSON  →  {input_path}")
    else:
        logger.info(f"Input mode: raw EHR text (MedGemma extraction)  →  {input_path}")

    try:
        assembler = ReportAssembler()

        if is_json:
            output_path = assembler.generate_and_save(
                patient_json_path=str(input_path),
                output_dir=args.output_dir,
            )
        else:
            output_path = assembler.generate_and_save_from_ehr(
                ehr_path=str(input_path),
                output_dir=args.output_dir,
                save_json_path=args.save_ehr_json,
            )

        logger.info("=" * 80)
        logger.info(f"✓ Report generated successfully: {output_path}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
