import io
import math
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None


class ImagePreprocessor:
    """Preprocesses handwritten note images for high-accuracy OCR."""

    def load_image(self, image_input) -> Image.Image:
        """Loads image from bytes or file path, supporting HEIC and auto EXIF orientation rotation."""
        if isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, str):
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise ValueError("Unsupported image input type")

        # Auto rotate based on EXIF tag
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        return img.convert("RGB")

    def detect_blur(self, image_np: np.ndarray) -> float:
        """
        Calculates Laplacian variance to quantify image blur.
        Lower values (<100) indicate blurry images.
        """
        if cv2 is None:
            return 100.0

        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def preprocess_for_ocr(self, img: Image.Image) -> Dict[str, Any]:
        """
        Performs full image preprocessing pipeline:
        - EXIF transpose & RGB conversion
        - Blur detection
        - Grayscale conversion
        - Deskewing
        - Denoising & Adaptive thresholding
        - Shadow removal
        """
        img_rgb = ImageOps.exif_transpose(img).convert("RGB")
        np_img = np.array(img_rgb)

        blur_score = self.detect_blur(np_img)
        is_blurry = blur_score < 80.0

        if cv2 is None:
            return {
                "pil_image": img_rgb,
                "processed_np": np_img,
                "blur_score": blur_score,
                "is_blurry": is_blurry,
            }

        # 1. Grayscale
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        # 2. Shadow removal / illumination normalization
        dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated, 21)
        diff_img = 255 - cv2.absdiff(gray, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        # 3. Deskewing using minimum area rectangle on text contour
        angle = 0.0
        try:
            thresh_for_angle = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            pts = np.column_stack(np.where(thresh_for_angle > 0))
            if len(pts) > 100:
                angle = cv2.minAreaRect(pts)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                elif angle > 45:
                    angle = 90 - angle

                if abs(angle) > 0.5 and abs(angle) < 45.0:
                    (h, w) = norm_img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    norm_img = cv2.warpAffine(norm_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            pass

        # 4. Denoising
        denoised = cv2.fastNlMeansDenoising(norm_img, h=10, searchWindowSize=21, templateWindowSize=7)

        # 5. Adaptive thresholding for binarization
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
        )

        processed_pil = Image.fromarray(binary)

        return {
            "pil_image": processed_pil,
            "processed_np": binary,
            "raw_grayscale": norm_img,
            "blur_score": blur_score,
            "is_blurry": is_blurry,
            "skew_angle": angle,
        }


image_preprocessor = ImagePreprocessor()
