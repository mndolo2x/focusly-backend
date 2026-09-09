import os
from typing import Optional
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image

class TextExtractor:
    """Utility class for PDF text extraction with PyMuPDF, pdfplumber, and pytesseract OCR fallback."""

    def extract_text_from_file(self, file_path: str) -> str:
        """Extracts text from PDF file or image."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".png", ".jpg", ".jpeg"]:
            return self._ocr_image(file_path)
        elif ext == ".pdf":
            return self._extract_from_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

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

    def _extract_from_pdf(self, file_path: str) -> str:
        """Attempts fast extraction via PyMuPDF, falls back to pdfplumber and OCR if text is scarce."""
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        except Exception:
            pass

        if len(text.strip()) > 50:
            return text.strip()

        # Fallback to pdfplumber
        try:
            with pdfplumber.open(file_path) as pdf:
                plumber_text = ""
                for page in pdf.pages:
                    extract = page.extract_text()
                    if extract:
                        plumber_text += extract + "\n"
                if len(plumber_text.strip()) > 50:
                    return plumber_text.strip()
        except Exception:
            pass

        # OCR fallback for scanned PDFs using PyMuPDF page rendering + Tesseract
        try:
            ocr_text = ""
            doc = fitz.open(file_path)
            for page in doc:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text += pytesseract.image_to_string(img) + "\n"
            doc.close()
            if ocr_text.strip():
                return ocr_text.strip()
        except Exception:
            pass

        return text.strip()

    def _ocr_image(self, image_path: str) -> str:
        """Extracts text from image file using Tesseract OCR."""
        try:
            img = Image.open(image_path)
            return pytesseract.image_to_string(img).strip()
        except Exception:
            return ""

text_extractor = TextExtractor()
