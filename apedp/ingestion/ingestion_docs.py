"""APEDP document ingestion module.

Loads local DOCX/PDF specifications and returns their text as a dict {name: text}.
Uses python-docx for DOCX reading and pypdf for PDF reading."""

import logging
import os
import xml.etree.ElementTree as ET  # nosec B405
import zipfile

logger = logging.getLogger(__name__)

try:
    import pypdf
except ImportError:
    pypdf = None  # type: ignore[assignment]


DEFAULT_FILES: list[str] = [
    'APEDP_Technical_Hurdles_and_Colab_Plan.pdf',
    'Architecture_Gap_Analysis_and_Baseline_Specification.pdf',
    'APEDP_Scientific_Discovery_Metrology_Architecture_Specification.docx',
    'APEDP_Scientific_Mathematical_Assurance_Specification.docx',
    'APEDP_NonTechnical_Requirements_Operating_Model.docx',
    'APEDP_Technical_Requirements_Specification.docx',
]


def extract_text_from_docx(fp: str) -> str:
    """Extract plain text from a DOCX file using standard library zipfile & xml."""
    try:
        with zipfile.ZipFile(fp) as z:
            xml_content = z.read('word/document.xml')
            # Input is a local user-supplied DOCX (same exposure as the python-docx
            # dependency); parse failures are contained by the handler below.
            tree = ET.fromstring(xml_content)  # noqa: S314  # nosec B314 - local DOCX, guarded below
            paragraphs = []
            for p_node in tree.iter():
                if p_node.tag.endswith('}p'):
                    p_text = ''.join(
                        t_node.text for t_node in p_node.iter()
                        if t_node.tag.endswith('}t') and t_node.text
                    )
                    if p_text.strip():
                        paragraphs.append(p_text.strip())
            return '\n'.join(paragraphs)
    except (zipfile.BadZipFile, ET.ParseError, KeyError, OSError, ValueError) as e:
        logger.warning("DOCX extraction failed for '%s' (%s)", fp, e)
        return ''


def extract_text_from_pdf(fp: str) -> str:
    """Extract plain text from a PDF file using pypdf."""
    if pypdf is None:
        return ''
    try:
        reader = pypdf.PdfReader(fp)
        return '\n'.join(page.extract_text() or '' for page in reader.pages)
    except (OSError, ValueError, AttributeError) as e:
        logger.warning("PDF extraction failed for '%s' (%s)", fp, e)
        return ''


def get_text(fp: str) -> str:
    """Extract plain text from a DOCX or PDF file."""
    if fp.endswith('.docx'):
        return extract_text_from_docx(fp)
    if fp.endswith('.pdf'):
        return extract_text_from_pdf(fp)
    return ''


def load_apedp_documents(base_path: str = '.') -> dict[str, str]:
    """Load all default APEDP documents from base_path.

    Returns {filename: text}. Missing files are skipped.
    """
    all_docs: dict[str, str] = {}
    for name in DEFAULT_FILES:
        fp = os.path.join(base_path, name)
        if os.path.exists(fp):
            text = get_text(fp)
            all_docs[name] = text
    return all_docs


if __name__ == '__main__':
    docs = load_apedp_documents('.')
    for k, v in docs.items():
        print(f"Loaded {k}: {len(v)} characters")
