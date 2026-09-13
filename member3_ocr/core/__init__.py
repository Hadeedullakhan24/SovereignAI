"""Member 3 Core Production Modules."""

from .ocr_pipeline import (
    OCRPipeline,
    PaddleOCRBackend,
    PaddleOCRModelConfig,
    OCRDocumentResult,
    OCRPageResult,
    TextBlock,
    BoundingBox,
    BackendCapabilities,
    BackendInfo,
)
from .image_preprocessing import (
    ImagePreprocessor,
    PreprocessingOptions,
    PreprocessingResult,
    preprocess_image,
    validate_image,
    estimate_skew_angle,
)
from .pdf_rendering import (
    render_pdf_pages,
    validate_pdf_path,
    get_pdf_page_count,
)
from .form_extractor import (
    FormKeyValueExtractor,
    FormExtractorConfig,
    extract_form_key_values,
)
from .table_extractor import (
    TableExtractor,
    TableExtractorConfig,
    extract_tables_from_blocks,
)
from .document_parser import (
    DocumentParser,
    DocumentParserError,
    AgentQueryableDocument,
    export_for_rag as export_document_for_rag,
    export_for_agent as export_document_for_agent,
)
from .drawing_analyzer import (
    DrawingAnalyzer,
    DrawingAnalyzerConfig,
    DrawingAnalysisRecord,
    AgentQueryableDrawing,
    DrawingAnalysisResult,
)
from .vision_pipeline import (
    VisionPipeline,
    VisionModelConfig,
    VisionResult,
    VisionAnalysisRecord,
    AgentQueryableVisionResult,
)
from .multimodal_processor import (
    MultimodalProcessor,
    MultimodalProcessingResult,
    RoutingDecision,
)
from .common_validators import (
    check_hallucinations,
    DRAWING_SYNONYMS,
    check_drawing_type_match,
    inspect_pid_outputs,
)

__all__ = [
    "OCRPipeline",
    "PaddleOCRBackend",
    "PaddleOCRModelConfig",
    "OCRDocumentResult",
    "OCRPageResult",
    "TextBlock",
    "BoundingBox",
    "BackendCapabilities",
    "BackendInfo",
    "ImagePreprocessor",
    "PreprocessingOptions",
    "PreprocessingResult",
    "preprocess_image",
    "validate_image",
    "estimate_skew_angle",
    "render_pdf_pages",
    "validate_pdf_path",
    "get_pdf_page_count",
    "FormKeyValueExtractor",
    "FormExtractorConfig",
    "extract_form_key_values",
    "TableExtractor",
    "TableExtractorConfig",
    "extract_tables_from_blocks",
    "DocumentParser",
    "DocumentParserError",
    "AgentQueryableDocument",
    "DrawingAnalyzer",
    "DrawingAnalyzerConfig",
    "DrawingAnalysisRecord",
    "AgentQueryableDrawing",
    "DrawingAnalysisResult",
    "VisionPipeline",
    "VisionModelConfig",
    "VisionResult",
    "VisionAnalysisRecord",
    "AgentQueryableVisionResult",
    "MultimodalProcessor",
    "MultimodalProcessingResult",
    "RoutingDecision",
    "check_hallucinations",
    "DRAWING_SYNONYMS",
    "check_drawing_type_match",
    "inspect_pid_outputs",
]
