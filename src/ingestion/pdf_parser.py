import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from loguru import logger


class ParsedDocument(BaseModel):
    doc_id: str
    title: str
    content: str
    page_count: int
    source: str = "pdf"
    metadata: dict = {}


class PDFParser:
    """
    Parses clinical guideline PDFs using PyMuPDF.

    Where to get clinical guidelines (all free):
      ACC/AHA : https://www.acc.org/guidelines
      NICE    : https://www.nice.org.uk/guidance
      WHO     : https://www.who.int/publications

    Place downloaded PDFs in data/guidelines/ and run the pipeline.
    """

    def __init__(self, pdf_dir: str = "data/guidelines"):
        self.pdf_dir = Path(pdf_dir)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PDF parser ready | dir={self.pdf_dir}")

    def parse_file(self, file_path: str) -> Optional[ParsedDocument]:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Not found: {file_path}")
            return None

        try:
            doc = fitz.open(str(path))
            pages_text = []

            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    pages_text.append(f"[Page {page_num + 1}]\n{text.strip()}")

            doc.close()
            full_content = "\n\n".join(pages_text)

            if not full_content.strip():
                logger.warning(f"No text in {path.name} — may be a scanned PDF")
                return None

            title = path.stem.replace("_", " ").replace("-", " ").title()
            logger.info(f"Parsed: {path.name} | {len(doc)} pages")

            return ParsedDocument(
                doc_id=f"pdf_{path.stem}",
                title=title,
                content=full_content,
                page_count=len(doc),
                metadata={"filename": path.name, "filepath": str(path.absolute())}
            )

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return None

    def parse_directory(self) -> List[ParsedDocument]:
        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDFs in {self.pdf_dir}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF(s)")
        docs = [self.parse_file(str(p)) for p in pdf_files]
        valid = [d for d in docs if d is not None]
        logger.success(f"Parsed {len(valid)}/{len(pdf_files)} PDFs")
        return valid
