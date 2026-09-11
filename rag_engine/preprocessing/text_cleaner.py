"""Section and string level cleaning coordinator."""

from __future__ import annotations

from rag_engine.preprocessing.engineering_token_protector import EngineeringTokenProtector
from rag_engine.preprocessing.normalizers import (
    BulletNormalizer,
    EncodingNormalizer,
    ListNormalizer,
    UnicodeNormalizer,
)
from rag_engine.preprocessing.whitespace_cleaner import WhitespaceCleaner
from rag_engine.schemas.parsed_document import Section


class TextCleaner:
    """Coordinates text normalization across sections, preserving engineering tokens."""

    def __init__(self, protector: EngineeringTokenProtector | None = None) -> None:
        self.protector = protector or EngineeringTokenProtector()

    def clean_text(
        self,
        text: str,
        protect_tokens: bool = True,
        apply_paragraph_reconstruction: bool = True,
        apply_hyphen_repair: bool = True,
    ) -> tuple[str, list[str]]:
        """Run text cleaning pipeline on a raw text string.

        Returns (cleaned_text, protected_tokens_found).
        """
        if not text:
            return "", []

        # 1. Mask engineering tokens if enabled
        masked_text = text
        token_map: dict[str, str] = {}
        tokens_found: list[str] = []

        if protect_tokens:
            tokens_found = self.protector.extract_tokens(text)
            masked_text, token_map = self.protector.mask(text)

        # Stage 1: Unicode normalization (NFKC)
        t1 = UnicodeNormalizer.normalize(masked_text)

        # Stage 2: Encoding normalization
        t2 = EncodingNormalizer.normalize(t1)

        # Stage 4: Line ending normalization
        t4 = WhitespaceCleaner.normalize_line_endings(t2)

        # Stage 6: Hyphenated word reconstruction (before whitespace collapse)
        t6 = (
            WhitespaceCleaner.reconstruct_hyphenated_words(t4)
            if apply_hyphen_repair
            else t4
        )

        # Stage 5: Broken paragraph reconstruction
        t5 = (
            WhitespaceCleaner.reconstruct_broken_paragraphs(t6)
            if apply_paragraph_reconstruction
            else t6
        )

        # Stage 10: Bullet normalization
        t10 = BulletNormalizer.normalize(t5)

        # Stage 11: List normalization
        t11 = ListNormalizer.normalize(t10)

        # Stage 3: Whitespace normalization
        t3 = WhitespaceCleaner.clean_whitespace(t11)

        # Stage 12: Unmask protected engineering tokens
        final_text = t3
        if protect_tokens and token_map:
            final_text = self.protector.unmask(t3, token_map)

        return final_text, tokens_found

    def clean_section(self, section: Section) -> tuple[Section, list[str]]:
        """Clean section content and re-derive paragraphs and lists."""
        cleaned_content, tokens = self.clean_text(section.content)

        # Clean paragraphs
        cleaned_paras: list[str] = []
        for p in section.paragraphs:
            cleaned_p, _ = self.clean_text(p, protect_tokens=True)
            if cleaned_p:
                cleaned_paras.append(cleaned_p)

        # Clean bullets
        cleaned_bullets: list[str] = []
        for b in section.bullet_points:
            cleaned_b, _ = self.clean_text(b, protect_tokens=True)
            if cleaned_b:
                cleaned_bullets.append(cleaned_b)

        # Clean numbered items
        cleaned_numbered: list[str] = []
        for n in section.numbered_items:
            cleaned_n, _ = self.clean_text(n, protect_tokens=True)
            if cleaned_n:
                cleaned_numbered.append(cleaned_n)

        # Clean child subsections recursively
        cleaned_subsections: list[Section] = []
        for sub in section.subsections:
            cleaned_sub, sub_tokens = self.clean_section(sub)
            cleaned_subsections.append(cleaned_sub)
            tokens.extend(sub_tokens)

        new_section = section.model_copy(
            update={
                "content": cleaned_content,
                "normalized_text": cleaned_content,
                "paragraphs": cleaned_paras,
                "bullet_points": cleaned_bullets,
                "numbered_items": cleaned_numbered,
                "subsections": cleaned_subsections,
            }
        )
        return new_section, tokens
