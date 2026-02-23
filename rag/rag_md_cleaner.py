"""
rag_md_cleaner.py

Markdown cleaner optimized for PDF -> Markdown extraction
(docling style) intended for RAG ingestion.

Primary concerns addressed:
- HTML comments like <!-- image -->
- Ligature /uniFB01 /uniFB02 /uniFB03 artifacts and unicode normalization
- Broken hyphen spacing "immune - related" => "immune-related"
- Standalone pipe lines " | "
- Tables (optional removal)
- Figure captions (optional removal)
- Reference removal: 
- Remove common metadata (Funding, Author contributions, Conflict of interest, Publisher's note)
"""

import re
import unicodedata
from typing import Optional

def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    # common PDF extraction broken ligature tokens and odd markers
    ligature_map = {
        "/uniFB01": "fi",
        "/uniFB02": "fl",
        "/uniFB03": "ffi",
        "/uniFB04": "ffl",
        "\ufb01": "fi",  
        "\ufb02": "fl",  
    }
    for k, v in ligature_map.items():
        text = text.replace(k, v)
    return text

def is_reference_like_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False

    # common signals of a reference line
    patterns = [
        r"\bdoi\s*:\s*10\.",            
        r"\bdoi\.?/?10\.",             
        r"\(\s*\d{4}\s*\)",           
        r"^\s*\d+\.\s+",               
        r"\bet al\.",                
        r"\bPMID\b|\bPMC\b",           
    ]
    for p in patterns:
        if re.search(p, s, flags=re.IGNORECASE):
            return True

    comma_count = s.count(",")
    if comma_count >= 3 and len(s) < 300:
        # heuristic: author lines usually have at least one capitalized surname-like token
        if re.search(r"[A-Z][a-z]{2,}\s+[A-Z]\b", s) or re.search(r"[A-Z][a-z]{2,},\s+[A-Z]", s):
            return True

    # journal-like end pattern: volume:pages or year;volume:pages
    if re.search(r"\d{4}\).*?\d{1,4}[:](\d|–|-)", s) or re.search(r"\b\d{1,4}:\d{1,4}\b", s):
        return True

    return False



def remove_references_section(
    text: str,
    mode: str = "conservative",
    consecutive_threshold_conservative: int = 5,
    consecutive_threshold_aggressive: int = 3,
    window_tail_fraction: float = 0.35,
) -> str:
    """
    Remove references from text.

    Parameters
    ----------
    text : str
        Raw markdown text
    mode : str
        "conservative" (safer, fewer false positives) or "aggressive" (more likely to remove refs)
    consecutive_threshold_conservative : int
        number of consecutive ref-like lines required for conservative mode
    consecutive_threshold_aggressive : int
        threshold for aggressive mode
    window_tail_fraction : float
        fraction of document considered the "tail" for additional detection

    Returns
    -------
    str
        Text trimmed before detected references (or original if none detected)
    """

    # 1) Try explicit headers
    header_regexes = [
        r"\n##\s*REFERENCES\b", r"\n##\s*References\b", r"\nREFERENCES\b", r"\nReferences\b"
    ]
    for hdr in header_regexes:
        m = re.search(hdr, text, flags=re.IGNORECASE)
        if m:
            return text[: m.start()]

    # 2) Heuristic scan for consecutive reference-like lines
    lines = text.splitlines()
    threshold = consecutive_threshold_conservative if mode == "conservative" else consecutive_threshold_aggressive

    consecutive = 0
    for idx, line in enumerate(lines):
        if is_reference_like_line(line):
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= threshold:
            # find the first line index where this consecutive block started
            start_idx = idx - consecutive + 1
            # return up to before that block
            return "\n".join(lines[:start_idx]).rstrip()

    # 3) Tail-window detection: maybe refs are at the end but not consecutive enough earlier
    n_lines = len(lines)
    tail_start = int(n_lines * (1.0 - window_tail_fraction))
    tail = lines[tail_start:]
    # compute fraction of ref-like lines in tail
    ref_like_count = sum(1 for L in tail if is_reference_like_line(L))
    if len(tail) > 10 and (ref_like_count / len(tail)) > 0.25:
        # find first ref-like line in tail and cut before it
        for i, L in enumerate(tail):
            if is_reference_like_line(L):
                return "\n".join(lines[: tail_start + i]).rstrip()

    # 4) No reliable reference block found -> return original
    return text



def remove_metadata_sections(text: str) -> str:
    """
    Remove common metadata sections by header names.
    Cuts text at the first occurrence of any of these headers.
    """
    meta_headers = [
        "AUTHOR CONTRIBUTIONS",
        "Author contributions",
        "FUNDING",
        "Funding",
        "CONFLICT OF INTEREST",
        "CONFLICT OF INTEREST STATEMENT",
        "Conflict of interest",
        "CONFLICT OF INTEREST STATEMENT",
        "Publisher's note",
        "Publisher ' s note",
        "Publisher?s note",
        "ACKNOWLEDGMENTS",
        "Acknowledgments",
        "DATA AVAILABILITY STATEMENT",
        "Data availability statement",
        "ORCID",
        "DATA AVAILABILITY",
        "Data Availability",
    ]
    pattern = r"\n(?:" + "|".join([re.escape(h) for h in meta_headers]) + r")\b"
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if m:
        return text[: m.start()].rstrip()
    return text


# ---------------------------
# Main cleaning function
# ---------------------------
def clean_markdown_for_rag(
    markdown_text: str,
    remove_tables: bool = False,
    remove_figures: bool = True,
    remove_references: bool = True,
    reference_mode: str = "conservative", 
    remove_metadata: bool = True,
    collapse_multiblank: bool = True,
) -> str:
    """
    Clean markdown text extracted from PDFs to produce RAG-friendly text.

    Parameters
    ----------
    markdown_text : str
        Raw markdown
    remove_tables : bool
        Remove markdown tables (default False). 
    remove_figures : bool
        Remove figure captions / figure blocks (default True)
    remove_references : bool
        Attempt to remove references (default True)
    reference_mode : str
        'conservative' or 'aggressive'
    remove_metadata : bool
        Remove author contributions/funding/conflict etc (default True)
    collapse_multiblank : bool
        Collapse >2 newlines to 2 newlines (default True)

    Returns
    -------
    str
        Cleaned markdown
    """
    text = markdown_text

    # normalize unicode & ligatures
    text = normalize_unicode(text)

    # remove HTML comments like <!-- image -->
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # remove explicit image placeholders (variants)
    text = re.sub(r"<\s*--\s*image\s*--\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[image:\s*.*?\]", "", text, flags=re.IGNORECASE)

    # remove standalone pipes lines
    text = re.sub(r"^\s*\|\s*$", "", text, flags=re.MULTILINE)

    # Optionally remove markdown tables (entire blocks). Conservative removal:
    if remove_tables:
        # Remove contiguous table-like blocks beginning and ending with pipes or table dividers
        text = re.sub(
            r"\n(?:\|[^\n]*\|\s*\n(?:\|[-:\s|]+\|\s*\n)?(?:\|[^\n]*\|\s*\n)+)",
            "\n",
            text,
            flags=re.DOTALL,
        )
        # Also remove inline table fragments
        text = re.sub(r"\n\|[-:\s|]+\|\n", "\n", text)

    else:
        # clean duplicate pipes
        text = re.sub(r"\|{2,}", "|", text)

    # Remove figure captions/blocks like FIGURE 1 ... Figure 1: ... or Fig. 1
    if remove_figures:
        text = re.sub(r"(?is)\bFIGURE\s*\d+[:.\s\S]*?(?=\n##|\n[A-Z]{2,}|$)", "", text)
        text = re.sub(r"(?is)\bFigure\s*\d+[:.\s\S]*?(?=\n##|\n[A-Z]{2,}|$)", "", text)
        text = re.sub(r"(?is)\bFig\.\s*\d+[:.\s\S]*?(?=\n##|\n[A-Z]{2,}|$)", "", text)

    # Fix hyphen spacing artifacts: "immune - related" -> "immune-related"
    text = re.sub(r"\s-\s+", "-", text)

    # Fix spaced punctuation: "word ," -> "word,"
    text = re.sub(r"\s+([,.;:])", r"\1", text)

    # Remove common publisher footer / 'Downloaded from...' blocks by looking for typical phrases
    text = re.sub(
        r"Downloaded from .*?Terms and Conditions.*?(?:\n|$)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # generic residual footer lines with DOI-like trailing
    text = re.sub(r"\n\d{6,}x?,?\s*\d{4}.*$", "", text, flags=re.DOTALL)

    # Remove references (robust)
    if remove_references:
        text = remove_references_section(text, mode=reference_mode)

    # Remove metadata sections (Funding, Author contributions, Conflicts, ORCID, etc.)
    if remove_metadata:
        text = remove_metadata_sections(text)

    # Collapse excessive blank lines
    if collapse_multiblank:
        text = re.sub(r"\n{3,}", "\n\n", text)

    # Trim leading/trailing whitespace
    text = text.strip()

    return text