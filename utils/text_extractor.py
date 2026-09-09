import os
from typing import Optional, Tuple
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image

class TextExtractor:
    """Utility class for PDF text extraction with PyMuPDF, pdfplumber, and pytesseract OCR fallback."""

    def extract_text_from_file(self, file_path: str) -> str:
        """Extracts text from PDF file or image."""
        text, _ = self.extract_text_and_page_count(file_path)
        return text

    def extract_text_and_page_count(self, file_path: str) -> Tuple[str, int]:
        """
        Extracts text and page count from PDF or image file.
        Uses PyMuPDF first, Tesseract OCR for scanned pages, and pdfplumber fallback.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".png", ".jpg", ".jpeg"]:
            return self._ocr_image(file_path), 1

        if ext == ".pdf":
            return self._extract_from_pdf_with_tracking(file_path)

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            return content, 1

    def get_page_count(self, file_path: str) -> int:
        """Returns total page count of PDF file."""
        if not os.path.exists(file_path):
            return 0
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            try:
                doc = fitz.open(file_path)
                count = len(doc)
                doc.close()
                return count
            except Exception:
                return 1
        return 1

    def _extract_from_pdf_with_tracking(self, file_path: str) -> Tuple[str, int]:
        """Extracts text with PyMuPDF page tracking, OCR fallback for scanned pages, and pdfplumber fallback."""
        text = ""
        page_count = 0
        pymupdf_success = False

        # Attempt PyMuPDF primary extraction with page tracking
        try:
            doc = fitz.open(file_path)
            page_count = len(doc)
            pages_text = []

            for page_num in range(page_count):
                page = doc.load_page(page_num)
                page_str = page.get_text()
                if page_str.strip():
                    pages_text.append(f"--- Page {page_num + 1} ---\n{page_str}")

            doc.close()
            if pages_text:
                text = "\n\n".join(pages_text)
                pymupdf_success = True
        except Exception:
            pymupdf_success = False

        if pymupdf_success and len(text.strip()) > 50:
            return text.strip(), page_count

        # OCR fallback for scanned PDFs with PyMuPDF page rendering + Tesseract OCR
        try:
            doc = fitz.open(file_path)
            page_count = len(doc)
            ocr_pages = []

            for page_num in range(page_count):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_str = pytesseract.image_to_string(img)
                if ocr_str.strip():
                    ocr_pages.append(f"--- Page {page_num + 1} ---\n{ocr_str}")

            doc.close()
            if ocr_pages:
                ocr_text = "\n\n".join(ocr_pages)
                if len(ocr_text.strip()) > 20:
                    return ocr_text.strip(), page_count
        except Exception:
            pass

        # Fallback to pdfplumber
        try:
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                plumber_pages = []

                for page_num, page in enumerate(pdf.pages):
                    extract = page.extract_text()
                    if extract:
                        plumber_pages.append(f"--- Page {page_num + 1} ---\n{extract}")

                if plumber_pages:
                    plumber_text = "\n\n".join(plumber_pages)
                    if len(plumber_text.strip()) > 20:
                        return plumber_text.strip(), page_count
        except Exception:
            pass

        return text.strip(), page_count if page_count > 0 else 1

    def _ocr_image(self, image_path: str) -> str:
        """Extracts text from image file using Tesseract OCR."""
        try:
            img = Image.open(image_path)
            return pytesseract.image_to_string(img).strip()
        except Exception:
            return ""

text_extractor = TextExtractor()
