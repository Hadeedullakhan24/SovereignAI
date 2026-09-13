"""Member 3 OCR & Vision Intelligence Package.

Top-level entry point providing backward-compatible access to Core production modules.
"""

import sys

# Re-export public API
from .core import *

# Register submodule aliases for backward-compatibility with existing tests and imports
from .core import (
    ocr_pipeline,
    image_preprocessing,
    pdf_rendering,
    form_extractor,
    table_extractor,
    document_parser,
    drawing_analyzer,
    vision_pipeline,
    multimodal_processor,
    common_validators,
)

sys.modules["member3_ocr.ocr_pipeline"] = ocr_pipeline
sys.modules["member3_ocr.image_preprocessing"] = image_preprocessing
sys.modules["member3_ocr.pdf_rendering"] = pdf_rendering
sys.modules["member3_ocr.form_extractor"] = form_extractor
sys.modules["member3_ocr.table_extractor"] = table_extractor
sys.modules["member3_ocr.document_parser"] = document_parser
sys.modules["member3_ocr.drawing_analyzer"] = drawing_analyzer
sys.modules["member3_ocr.vision_pipeline"] = vision_pipeline
sys.modules["member3_ocr.multimodal_processor"] = multimodal_processor
sys.modules["member3_ocr.common_validators"] = common_validators

# Evaluation submodules backward-compatibility
from .evaluation.ocr import ocr_evaluator
sys.modules["member3_ocr.ocr_evaluator"] = ocr_evaluator
from .evaluation.document import evaluate_funsd_kv_f1
sys.modules["member3_ocr.evaluate_funsd_kv_f1"] = evaluate_funsd_kv_f1
from .evaluation.vision import evaluate_vision
sys.modules["member3_ocr.evaluate_vision"] = evaluate_vision
from .evaluation.drawing import evaluate_drawing_analyzer
sys.modules["member3_ocr.evaluate_drawing_analyzer"] = evaluate_drawing_analyzer
