"""Document ingestion: auto-rotate, normalize, deskew, split pages, store originals."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ExifTags

from graphocr.core.logging import get_logger
from graphocr.models.document import PageImage, RawDocument

logger = get_logger(__name__)

# Max dimension for OCR processing.
# 2500px gives best accuracy for handwritten Arabic text.
# Reduce to 1500 for faster CPU processing (9x speedup, slight quality loss).
MAX_SIDE = 2500


def load_document(
    doc: RawDocument,
    output_dir: str,
    max_side: int = MAX_SIDE,
) -> list[PageImage]:
    """Load a document and split into normalized page images.

    Pipeline: EXIF rotation fix -> orientation detect -> resize -> deskew -> CLAHE

    Args:
        doc: Raw document metadata.
        output_dir: Directory to write page images.
        max_side: Maximum dimension (pixels) to prevent OOM.

    Returns:
        List of PageImage objects with paths to normalized images.
    """
    source = Path(doc.source_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if doc.file_format == "pdf":
        pil_images = convert_from_path(str(source), dpi=300)
    else:
        pil_images = [Image.open(source)]

    pages: list[PageImage] = []
    for idx, pil_img in enumerate(pil_images):
        page_num = idx + 1

        # Step 1: Fix EXIF rotation (phone cameras embed rotation in metadata)
        pil_img = _fix_exif_rotation(pil_img)

        # Step 2: Detect and correct 90/180/270 rotation
        pil_img = _correct_orientation(pil_img)

        # Step 3: Resize if too large
        if max(pil_img.size) > max_side:
            ratio = max_side / max(pil_img.size)
            new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
            logger.info("image_resized", page=page_num, size=new_size)

        img_array = np.array(pil_img)

        # Step 4: Normalize (deskew + contrast)
        img_array = _normalize_image(img_array)

        # Save
        page_path = out / f"{doc.document_id}_page_{page_num}.png"
        cv2.imwrite(str(page_path), img_array)

        h, w = img_array.shape[:2]
        pages.append(
            PageImage(
                document_id=doc.document_id,
                page_number=page_num,
                image_path=str(page_path),
                width_px=w,
                height_px=h,
                dpi=300,
                is_deskewed=True,
                is_contrast_enhanced=True,
            )
        )

    logger.info("ingested_document", document_id=doc.document_id, pages=len(pages))
    return pages


def _fix_exif_rotation(img: Image.Image) -> Image.Image:
    """Fix rotation from EXIF metadata (common with phone camera photos).

    Phone cameras store orientation in EXIF tags rather than rotating
    the actual pixel data. This applies the EXIF rotation so the image
    pixels match what a human would see.
    """
    try:
        exif = img.getexif()
        if not exif:
            return img

        # EXIF tag 274 = Orientation
        orientation = exif.get(274)
        if orientation is None:
            return img

        rotations = {
            3: 180,
            6: 270,
            8: 90,
        }

        if orientation in rotations:
            angle = rotations[orientation]
            img = img.rotate(angle, expand=True)
            logger.info("exif_rotation_fixed", angle=angle, orientation=orientation)

    except Exception:
        pass  # No EXIF or corrupt — skip silently

    return img


def _correct_orientation(img: Image.Image) -> Image.Image:
    """Detect and correct 90/180/270 degree rotation.

    Uses text line detection: in a correctly oriented document, text lines
    should be roughly horizontal (wider than tall). If text lines are
    vertical, the image is rotated 90 or 270 degrees.

    Strategy:
    1. Convert to grayscale and threshold
    2. Find contours (text regions)
    3. Check aspect ratios of bounding rectangles
    4. If most regions are taller than wide → image is sideways
    5. Use the dominant angle from minAreaRect to determine 90 vs 270
    """
    img_array = np.array(img.convert("L"))

    # Threshold
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate to connect text into line-level blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 3:
        return img

    # Analyze bounding rectangles
    horizontal_count = 0
    vertical_count = 0
    angles = []

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (_, (w, h), angle) = rect

        # Skip tiny contours
        if w * h < 500:
            continue

        if w > h:
            horizontal_count += 1
        else:
            vertical_count += 1
            angles.append(angle)

    total = horizontal_count + vertical_count
    if total < 3:
        return img

    vertical_ratio = vertical_count / total

    # If >60% of text regions are vertical, image is sideways
    if vertical_ratio > 0.6:
        # Determine direction: check if image is portrait (h > w) already
        # and whether text runs top-to-bottom or bottom-to-top
        h_img, w_img = img_array.shape

        if w_img > h_img:
            # Landscape image with vertical text → rotate 90° CW
            rotation = 90
        else:
            # Portrait image with vertical text → rotate 90° CCW
            rotation = 270

        img = img.rotate(rotation, expand=True)
        logger.info("orientation_corrected", rotation=rotation, vertical_ratio=f"{vertical_ratio:.2f}")

    # Check for upside-down (180°): look at text density in top vs bottom half
    elif _is_upside_down(img_array):
        img = img.rotate(180, expand=True)
        logger.info("orientation_corrected", rotation=180, reason="upside_down")

    return img


def _is_upside_down(gray: np.ndarray) -> bool:
    """Detect if a document is upside down.

    Heuristic: In a correctly oriented document, the top region typically
    has header text (more ink density) than the very bottom. Also, text
    baselines create more dark pixels in the lower part of each text line.

    We compare the density of dark pixels in the top 15% vs bottom 15%.
    Headers/logos at the top should have higher density.
    """
    h = gray.shape[0]
    top_strip = gray[: int(h * 0.15), :]
    bottom_strip = gray[int(h * 0.85) :, :]

    top_dark = np.sum(top_strip < 128)
    bottom_dark = np.sum(bottom_strip < 128)

    # If bottom has significantly more dark pixels than top, likely upside down
    # But only if the difference is substantial
    if bottom_dark > 0 and top_dark > 0:
        ratio = bottom_dark / top_dark
        return ratio > 3.0  # Bottom has 3x more ink than top

    return False


def _normalize_image(img: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline for Arabic medical documents.

    Order is critical — derived from Invizo (2025), Arabic OCR surveys,
    and PaddleOCR best practices:

      1. Stamp suppression (while still in color)
      2. Grayscale conversion
      3. Shadow/lighting correction
      4. Mild denoising (preserves Arabic thin strokes + dots)
      5. Contrast enhancement (CLAHE)
      6. Deskew (projection profile — more robust for Arabic than Hough)

    Output is grayscale-in-BGR (3 channels) for PaddleOCR.
    Do NOT binarize — PaddleOCR uses gradient info internally.
    """
    # Step 1: Stamp suppression while we still have color
    if len(img.shape) == 3:
        gray = _suppress_stamps(img)
    else:
        gray = img.copy()

    # Step 2: Shadow and uneven lighting correction
    gray = _correct_lighting(gray)

    # Step 3: Mild denoising — preserves Arabic thin strokes and diacritical dots
    # FNLM with h=3 is safe for Arabic; h>5 starts destroying dot connections
    gray = cv2.fastNlMeansDenoising(gray, None, h=3, templateWindowSize=7, searchWindowSize=21)

    # Step 4: Contrast enhancement (CLAHE)
    # clipLimit=2.5 tuned for faded medical prescriptions (up from 2.0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Step 5: Deskew via projection profile (more robust for Arabic than Hough)
    gray = _deskew_projection(gray)

    # Return as BGR for PaddleOCR (expects 3-channel)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _suppress_stamps(img_bgr: np.ndarray) -> np.ndarray:
    """Suppress colored stamps/seals while preserving dark text.

    Medical prescriptions have red/blue/green pharmacy stamps overlapping
    handwritten text. This detects highly saturated colored pixels (stamps)
    and fades them, preserving underlying dark ink strokes.

    Works because: text ink is low-saturation (black/dark gray),
    stamps are high-saturation colors.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Detect stamp pixels: high saturation AND not too dark (not black text)
    stamp_mask = (saturation > 80) & (value > 50)

    # Also catch common stamp colors specifically
    hue = hsv[:, :, 0]
    red_mask = ((hue < 10) | (hue > 170)) & (saturation > 60)
    blue_mask = ((hue > 100) & (hue < 130)) & (saturation > 60)
    green_mask = ((hue > 35) & (hue < 85)) & (saturation > 60)

    combined = stamp_mask | red_mask | blue_mask | green_mask

    # Dilate mask slightly to cover stamp edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.dilate(combined.astype(np.uint8), kernel, iterations=1).astype(bool)

    # Conservative: fade stamps rather than erase — preserves underlying dark strokes
    result = gray.copy()
    result[combined] = np.minimum(
        result[combined].astype(np.int16) + 80, 255
    ).astype(np.uint8)

    logger.info("stamps_suppressed", pixels_affected=int(np.sum(combined)))
    return result


def _correct_lighting(gray: np.ndarray) -> np.ndarray:
    """Remove uneven lighting via morphological background estimation.

    Estimates the background illumination using a large morphological
    closing, then divides it out. This flattens lighting gradients from
    phone camera captures.

    The kernel size is ~6% of the smaller image dimension — large enough
    to span text strokes but captures illumination gradients.
    """
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    # Kernel must be larger than any text stroke
    h, w = gray.shape
    ksize = max(51, int(min(h, w) * 0.06)) | 1  # ensure odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    background = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    # Division-based correction: normalizes illumination
    bg_float = background.astype(np.float32) + 1e-6
    corrected = (gray.astype(np.float32) / bg_float * 255.0)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _deskew_projection(gray: np.ndarray) -> np.ndarray:
    """Deskew using projection profile analysis.

    More robust than Hough lines for Arabic text because Arabic has
    more vertical strokes that confuse Hough. Projection profile directly
    measures 'crispness' of horizontal text lines.

    Tries angles from -5 to +5 degrees and picks the one that maximizes
    the variance of the horizontal projection (sharpest text lines).
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_angle = 0.0
    best_score = 0.0
    h, w = binary.shape
    center = (w // 2, h // 2)

    # Search -5 to +5 degrees in 0.2-degree steps
    for angle_10x in range(-50, 51, 2):
        angle = angle_10x / 10.0
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST)

        profile = np.sum(rotated, axis=1)
        score = float(np.var(profile))

        if score > best_score:
            best_score = score
            best_angle = angle

    if abs(best_angle) < 0.3:
        return gray

    logger.info("deskew_applied", angle=best_angle)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
