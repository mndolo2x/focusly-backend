import base64
import io
import re
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError:
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None

from services.gemini_service import gemini_service
from utils.image_preprocessor import image_preprocessor


class HandwritingOCR:
    """
    3-Tier Handwriting OCR Engine:
    - Tier 1: Local Ollama llama3.2-vision:11b for high accuracy handwriting transcription + [DIAGRAM: ...] tagging.
    - Tier 2: TrOCR / EasyOCR fallback for local neural OCR.
    - Tier 3: Tesseract OCR with --psm 6 (Assume a single uniform block of text).
    """

    def __init__(self):
        self._easyocr_reader = None
        self._trocr_processor = None
        self._trocr_model = None

    def _get_easyocr_reader(self):
        if self._easyocr_reader is None and easyocr is not None:
            try:
                self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
            except Exception:
                pass
        return self._easyocr_reader

    def _get_trocr_components(self):
        if self._trocr_processor is None and TrOCRProcessor is not None:
            try:
                self._trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
                self._trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            except Exception:
                pass
        return self._trocr_processor, self._trocr_model

    def encode_image_base64(self, img: Image.Image) -> str:
        """Converts PIL image to base64 JPEG string for vision model input."""
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    async def tier1_vision_llm(self, img: Image.Image) -> Optional[Dict[str, Any]]:
        """
        Tier 1: Query Gemini Vision model.
        Asks vision model to transcribe handwritten text and extract diagrams with [DIAGRAM: ...] tags.
        """
        prompt = (
            "Transcribe all handwritten text in this image accurately into clean Markdown text.\n"
            "If you see any drawn diagrams, flowcharts, or formulas, represent them inline using structured text tags like:\n"
            "[DIAGRAM: Description of diagram, elements, labels, and arrows]\n"
            "Do not output conversational filler. Provide only the transcribed note content."
        )

        try:
            response = await gemini_service.generate_with_image(
                prompt=prompt,
                image_input=img
            )
            if response and response.get("response"):
                text = response["response"].strip()
                if len(text) > 10:
                    return {
                        "text": text,
                        "confidence": 0.95,
                        "tier": "gemini-vision",
                        "diagrams_found": bool(re.search(r'\[DIAGRAM:.*?\]', text, re.IGNORECASE))
                    }
        except Exception:
            pass
        return None

    def tier2_neural_ocr(self, img: Image.Image) -> Optional[Dict[str, Any]]:
        """
        Tier 2: TrOCR (microsoft/trocr-base-handwritten) with EasyOCR fallback.
        """
        processor, model = self._get_trocr_components()
        if processor is not None and model is not None:
            try:
                pixel_values = processor(images=img.convert("RGB"), return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                if generated_text:
                    return {
                        "text": generated_text,
                        "confidence": 0.85,
                        "tier": "trocr-handwritten",
                        "diagrams_found": False
                    }
            except Exception:
                pass

        reader = self._get_easyocr_reader()
        if not reader:
            return None

        try:
            import numpy as np
            np_img = np.array(img)
            results = reader.readtext(np_img)
            if not results:
                return None

            lines = []
            conf_scores = []
            for bbox, text, prob in results:
                lines.append(text)
                conf_scores.append(prob)

            full_text = "\n".join(lines)
            avg_conf = float(sum(conf_scores) / len(conf_scores)) if conf_scores else 0.5

            return {
                "text": full_text,
                "confidence": round(avg_conf, 2),
                "tier": "easyocr",
                "diagrams_found": False
            }
        except Exception:
            return None

    def tier3_tesseract(self, img: Image.Image) -> Dict[str, Any]:
        """
        Tier 3: Tesseract OCR with --psm 6 (Assume single uniform block of text).
        """
        if pytesseract is None:
            return {
                "text": "",
                "confidence": 0.0,
                "tier": "tesseract_failed",
                "diagrams_found": False
            }

        try:
            # Custom config for uniform handwriting block
            config = "--psm 6"
            text = pytesseract.image_to_string(img, config=config).strip()

            # Confidence estimation from data
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
            confs = [float(c) for c in data.get("conf", []) if c != "-1" and str(c).replace(".", "", 1).isdigit()]
            avg_conf = (sum(confs) / len(confs)) / 100.0 if confs else 0.4

            return {
                "text": text,
                "confidence": round(avg_conf, 2),
                "tier": "tesseract",
                "diagrams_found": False
            }
        except Exception as e:
            return {
                "text": f"[OCR Error: {str(e)}]",
                "confidence": 0.0,
                "tier": "tesseract_error",
                "diagrams_found": False
            }

    def extract_diagram_descriptions(self, text: str) -> List[str]:
        """Parses inline [DIAGRAM: ...] tags from OCR transcription text."""
        return re.findall(r'\[DIAGRAM:\s*(.*?)\]', text, re.IGNORECASE | re.DOTALL)

    async def transcribe_handwritten_image(
        self, image_input: Any, page_num: int = 1
    ) -> Dict[str, Any]:
        """
        Runs full preprocessor + 3-tier OCR workflow on an image input.
        Returns extracted text, confidence score, diagrams, and low-confidence warning if needed.
        """
        pil_img = image_preprocessor.load_image(image_input)
        prep_res = image_preprocessor.preprocess_for_ocr(pil_img)
        processed_pil = prep_res["pil_image"]
        is_blurry = prep_res["is_blurry"]

        # Attempt Tier 1: llama3.2-vision
        t1_res = await self.tier1_vision_llm(pil_img)
        if t1_res and t1_res.get("text"):
            extracted_text = t1_res["text"]
            confidence = t1_res["confidence"]
            ocr_tier = t1_res["tier"]
        else:
            # Attempt Tier 2: TrOCR / EasyOCR
            t2_res = self.tier2_neural_ocr(processed_pil)
            if t2_res and t2_res.get("text") and len(t2_res["text"].strip()) > 5:
                extracted_text = t2_res["text"]
                confidence = t2_res["confidence"]
                ocr_tier = t2_res["tier"]
            else:
                # Attempt Tier 3: Tesseract --psm 6
                t3_res = self.tier3_tesseract(processed_pil)
                extracted_text = t3_res["text"]
                confidence = t3_res["confidence"]
                ocr_tier = t3_res["tier"]

        # Reduce confidence if image is blurry
        if is_blurry:
            confidence = max(0.1, round(confidence - 0.25, 2))

        diagrams = self.extract_diagram_descriptions(extracted_text)
        is_low_confidence = confidence < 0.60

        return {
            "virtual_page": page_num,
            "text": f"--- Virtual Page {page_num} ---\n" + extracted_text,
            "raw_text": extracted_text,
            "confidence": confidence,
            "ocr_tier": ocr_tier,
            "diagrams": diagrams,
            "is_blurry": is_blurry,
            "is_low_confidence": is_low_confidence,
            "blur_score": prep_res.get("blur_score", 100.0)
        }


handwriting_ocr = HandwritingOCR()
